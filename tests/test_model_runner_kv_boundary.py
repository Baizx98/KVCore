from __future__ import annotations

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from kvcore.kv.kv_manager import KVManagerConfig
from kvcore.kv.single_type_kv_manager import KVLayerSpec
from kvcore.model.model_loader import ModelLoadConfig
from kvcore.model.model_runner import ModelRunner
from kvcore.model.models.llama3 import Llama3ForCausalLM
from kvcore.sched.scheduler import Scheduler
from kvcore.sched.utils import SchedulerConfig
from kvcore.utils.request import Request
from kvcore.utils.sampling_params import SamplingParams


def make_runner_with_tiny_model() -> ModelRunner:
    runner = ModelRunner(ModelLoadConfig(model="unused", device="cpu", attn_backend="torch_paged"))
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    runner.model = Llama3ForCausalLM(config, attn_backend="torch_paged")
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
