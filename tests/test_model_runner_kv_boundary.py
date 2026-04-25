from __future__ import annotations

import torch
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig

from kvcore.config import KVCoreConfig, ModelConfig
from kvcore.kv.kv_manager import KVManager, KVManagerConfig
from kvcore.kv.single_type_kv_manager import KVLayerSpec
from kvcore.model.model_loader import ModelLoadConfig
from kvcore.model.model_runner import ModelRunner
from kvcore.model.models.llama3 import Llama3ForCausalLM
from kvcore.sched.scheduler import Scheduler
from kvcore.sched.utils import ScheduledRequest, SchedulerConfig, SchedulerOutput
from kvcore.utils.request import Request
from kvcore.utils.sampling_params import SamplingParams


def make_runner_with_tiny_model() -> ModelRunner:
    runner = ModelRunner(ModelLoadConfig(model="unused", device="cpu", attn_backend="torch_paged"))
    hf_config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    runner.model = Llama3ForCausalLM(
        KVCoreConfig(
            model=ModelConfig(
                model="unused",
                attn_backend="torch_paged",
                hf_config=hf_config,
            )
        )
    )
    return runner


def make_kv_config(num_layers: int = 2) -> KVManagerConfig:
    return KVManagerConfig(
        num_gpu_blocks=16,
        max_model_len=16,
        layer_specs=tuple(
            KVLayerSpec(
                layer_idx=layer_idx,
                block_size=2,
                num_kv_heads=2,
                head_size=8,
                dtype=torch.float16,
            )
            for layer_idx in range(num_layers)
        ),
    )


def make_request(request_id: str = "req", token_ids: list[int] | None = None) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids or [1, 2, 3, 4, 5, 6],
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
    )


class SpyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, torch.Tensor | None]] = []

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "input_ids": input_ids,
                "positions": positions,
                "inputs_embeds": inputs_embeds,
            }
        )
        return torch.arange(8, dtype=torch.float32).view(2, 4)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((hidden_states.shape[0], 16), dtype=hidden_states.dtype)
        logits[:, 5] = 1.0
        return logits


def make_manual_scheduler_output() -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_requests=(
            ScheduledRequest(
                request_id="req",
                is_prefill=True,
                flat_start=0,
                flat_end=2,
                context_len=0,
                query_len=2,
                sample_index=1,
                should_sample=True,
            ),
        ),
        flat_input_token_ids=(7, 8),
        flat_positions=(0, 1),
        sampling_params=(SamplingParams(max_tokens=1, temperature=0.0),),
        num_scheduled_tokens=2,
        num_prefill_reqs=1,
        num_decode_reqs=0,
    )


def test_model_runner_initializes_global_kv_cache_tensor() -> None:
    runner = make_runner_with_tiny_model()

    kv_cache_tensor = runner.initialize_kv_cache(make_kv_config())

    assert runner.kv_cache_tensor is not None
    assert kv_cache_tensor is runner.kv_cache_tensor
    assert runner.kv_cache_tensor.shape == (2, 16, 2, 2, 8)


def test_model_runner_builds_paged_attention_metadata_from_scheduler_output() -> None:
    runner = make_runner_with_tiny_model()
    kv_config = make_kv_config()
    runner.initialize_kv_cache(kv_config)
    scheduler = Scheduler(
        kv_config,
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_scheduled_tokens=4),
    )

    scheduler.add_request(make_request())
    scheduler_output = scheduler.schedule()
    metadata = runner.build_attention_metadata(scheduler.kv_manager, scheduler_output)

    assert metadata.num_reqs == 1
    assert metadata.num_scheduled_tokens == 4
    assert metadata.query_start_loc.tolist() == [0, 4]
    assert metadata.context_lens.tolist() == [0]
    assert metadata.seq_lens.tolist() == [4]
    assert metadata.query_lens.tolist() == [4]
    assert metadata.flat_positions.tolist() == [0, 1, 2, 3]
    assert metadata.token_request_indices.tolist() == [0, 0, 0, 0]
    assert metadata.kv_cache_tensor is runner.kv_cache_tensor


def test_model_runner_prepares_model_input_from_scheduler_output_fixture() -> None:
    runner = make_runner_with_tiny_model()
    kv_config = make_kv_config()
    runner.initialize_kv_cache(kv_config)
    kv_manager = KVManager(kv_config)
    request = make_request(token_ids=[7, 8])
    kv_manager.allocate_slots(request, num_new_tokens=2)

    model_input = runner.prepare_model_input(make_manual_scheduler_output(), kv_manager)

    assert model_input.input_ids.tolist() == [7, 8]
    assert model_input.positions.tolist() == [0, 1]
    assert model_input.sample_indices == (1,)
    assert model_input.sampled_request_ids == ("req",)
    assert model_input.attn_metadata.kv_cache_tensor is runner.kv_cache_tensor


def test_model_runner_execute_model_uses_forward_context_not_model_kwargs() -> None:
    runner = ModelRunner(ModelLoadConfig(model="unused", device="cpu", attn_backend="torch_paged"))
    spy_model = SpyCausalLM()
    runner.model = spy_model
    kv_config = make_kv_config(num_layers=1)
    runner.initialize_kv_cache(kv_config)
    kv_manager = KVManager(kv_config)
    request = make_request(token_ids=[7, 8])
    kv_manager.allocate_slots(request, num_new_tokens=2)

    step_output = runner.execute_model(make_manual_scheduler_output(), kv_manager)

    assert len(spy_model.calls) == 1
    assert spy_model.calls[0]["input_ids"].tolist() == [7, 8]
    assert spy_model.calls[0]["positions"].tolist() == [0, 1]
    assert step_output.sampled_request_ids == ("req",)
    assert step_output.sampled_token_ids == (5,)


def test_model_runner_execute_model_samples_requested_positions() -> None:
    runner = make_runner_with_tiny_model()
    kv_config = make_kv_config()
    runner.initialize_kv_cache(kv_config)
    scheduler = Scheduler(
        kv_config,
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_scheduled_tokens=8),
    )

    scheduler.add_request(make_request(token_ids=[1, 2, 3, 4]))
    scheduler_output = scheduler.schedule()
    step_output = runner.execute_model(scheduler_output, scheduler.kv_manager)

    assert step_output.sampled_request_ids == ("req",)
    assert len(step_output.sampled_token_ids) == 1
