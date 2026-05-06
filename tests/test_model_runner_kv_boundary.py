from __future__ import annotations

import torch
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig

import kvcore.model.input_batch as input_batch_module
from kvcore.config import (
    CacheConfig,
    DeviceConfig,
    KVCoreConfig,
    ModelConfig,
    SchedulerConfig,
    SparseKVConfig,
)
from kvcore.kv.kv_manager import KVManagerConfig
from kvcore.kv.single_type_kv_manager import KVLayerSpec
from kvcore.kv.sparse import LayerSparsePlan, SparseKVPlan
from kvcore.model.model_runner import ModelRunner
from kvcore.model.models.llama3 import Llama3ForCausalLM
from kvcore.sched.scheduler import Scheduler
from kvcore.sched.utils import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from kvcore.utils.request import Request
from kvcore.utils.sampling_params import SamplingParams


def make_runner_with_tiny_model() -> ModelRunner:
    runner = ModelRunner(make_runner_config())
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
            model_config=ModelConfig(
                model="unused",
                attn_backend="torch_paged",
                hf_config=hf_config,
            ),
            device_config=DeviceConfig(device="cpu"),
        )
    )
    return runner


def make_runner_config() -> KVCoreConfig:
    return KVCoreConfig(
        model_config=ModelConfig(model="unused", attn_backend="torch_paged"),
        device_config=DeviceConfig(device="cpu"),
    )


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


def make_request(
    request_id: str = "req",
    token_ids: list[int] | None = None,
    max_tokens: int = 1,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids or [1, 2, 3, 4, 5, 6],
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
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
        scheduled_new_reqs=(
            NewRequestData(
                req_id="req",
                prompt_token_ids=(7, 8),
                sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
                block_ids=((1,), (1,)),
                num_computed_tokens=0,
            ),
        ),
        scheduled_cached_reqs=CachedRequestData.empty(),
        num_scheduled_tokens={"req": 2},
        total_num_scheduled_tokens=2,
        new_block_ids_to_zero=(1,),
    )


def test_model_runner_initializes_global_kv_cache_tensor() -> None:
    runner = make_runner_with_tiny_model()

    kv_cache_tensor = runner.initialize_kv_cache(make_kv_config())

    assert runner.kv_cache_tensor is not None
    assert kv_cache_tensor is runner.kv_cache_tensor
    assert runner.kv_cache_tensor.shape == (2, 16, 2, 2, 8)


def test_model_runner_cuda_profile_uses_peak_memory_budget(monkeypatch) -> None:
    kvcore_config = KVCoreConfig(
        model_config=ModelConfig(model="unused", attn_backend="torch_paged"),
        cache_config=CacheConfig(block_size=2, num_gpu_blocks=None),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_scheduled_tokens=4),
        device_config=DeviceConfig(device="cuda:0"),
    )
    runner = ModelRunner(kvcore_config)
    runner.model = SpyCausalLM()
    layer_specs = make_kv_config(num_layers=1).layer_specs
    cuda_calls: list[str] = []

    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device=None: (1_000_000_000, 10_000_000_000),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cuda_calls.append("empty_cache"))
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device=None: cuda_calls.append("reset_peak"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: cuda_calls.append("synchronize"),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 3_000_000)
    monkeypatch.setattr(
        runner,
        "_profile_dummy_forward",
        lambda **_kwargs: cuda_calls.append("dummy_forward"),
    )

    profile = runner.profile_run(
        layer_specs=layer_specs,
        block_size=2,
        max_model_len=16,
        requested_num_gpu_blocks=None,
        should_profile=True,
        gpu_memory_utilization=0.8,
    )

    bytes_per_block = runner._get_bytes_per_block(layer_specs[0], torch.float16)
    profile_kv_bytes = 3 * bytes_per_block
    expected_non_kv_peak = 3_000_000 - profile_kv_bytes
    expected_post_kv_cache_memory = runner._estimate_post_kv_cache_memory_bytes(
        layer_specs=layer_specs,
        max_model_len=16,
    )
    expected_available_for_kv = min(
        8_000_000_000 - expected_non_kv_peak,
        800_000_000 - expected_post_kv_cache_memory - 256 * 1024 * 1024,
    )
    expected_blocks = expected_available_for_kv // bytes_per_block
    assert profile.num_gpu_blocks == expected_blocks
    assert profile.available_memory_bytes == 1_000_000_000
    assert profile.memory_budget_bytes == expected_available_for_kv
    assert profile.peak_memory_bytes == 3_000_000
    assert profile.non_kv_peak_memory_bytes == expected_non_kv_peak
    assert "dummy_forward" in cuda_calls
    assert runner.kv_cache_tensor is None
    assert runner.input_batch is None
    assert runner.requests == {}


def test_model_runner_profile_state_can_be_cleared_before_real_kv_init() -> None:
    runner = make_runner_with_tiny_model()
    kv_config = make_kv_config()

    runner._profile_dummy_forward(
        layer_specs=kv_config.layer_specs,
        max_model_len=kv_config.max_model_len,
        profile_tokens=4,
        profile_num_gpu_blocks=3,
    )
    assert runner.kv_cache_tensor is not None
    assert runner.input_batch is not None
    assert runner.requests

    runner._clear_profile_state()
    assert runner.kv_cache_tensor is None
    assert runner.input_batch is None
    assert runner.requests == {}

    kv_cache_tensor = runner.initialize_kv_cache(kv_config)
    assert kv_cache_tensor.shape == (2, 16, 2, 2, 8)


def test_model_runner_builds_paged_attention_metadata_from_scheduler_output() -> None:
    runner = make_runner_with_tiny_model()
    kv_config = make_kv_config()
    runner.initialize_kv_cache(kv_config)
    scheduler = Scheduler(
        KVCoreConfig(
            model_config=ModelConfig(model="unused", attn_backend="torch_paged"),
            scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_scheduled_tokens=4),
            device_config=DeviceConfig(device="cpu"),
        ),
        kv_config,
    )

    scheduler.add_request(make_request())
    scheduler_output = scheduler.schedule()
    runner._update_states(scheduler_output)
    num_scheduled_tokens = tuple(
        scheduler_output.num_scheduled_tokens[req_id]
        for req_id in runner._require_input_batch().req_ids
    )
    (
        _logits_indices,
        _sampled_request_ids,
        _sampling_params,
        num_prefill_reqs,
        num_decode_reqs,
        max_query_len,
        max_seq_len,
    ) = runner._prepare_inputs(scheduler_output, num_scheduled_tokens)
    metadata = runner._build_attention_metadata(
        scheduler_output=scheduler_output,
        num_reqs=runner._require_input_batch().num_reqs,
        num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
        num_prefill_reqs=num_prefill_reqs,
        num_decode_reqs=num_decode_reqs,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
    )

    assert metadata.num_reqs == 1
    assert metadata.num_scheduled_tokens == 4
    assert metadata.query_start_loc.tolist() == [0, 4]
    assert metadata.context_lens.tolist() == [0]
    assert metadata.seq_lens.tolist() == [4]
    assert metadata.query_lens.tolist() == [4]
    assert metadata.flat_positions.tolist() == [0, 1, 2, 3]
    assert metadata.token_request_indices.tolist() == [0, 0, 0, 0]
    assert metadata.kv_cache_tensor is runner.kv_cache_tensor


def test_input_batch_metadata_wrapper_is_removed() -> None:
    assert not hasattr(input_batch_module, "InputBatchMetadata")


def test_model_runner_materializes_sparse_read_table_from_scheduler_plan() -> None:
    runner = make_runner_with_tiny_model()
    runner.initialize_kv_cache(make_kv_config())
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=(
            NewRequestData(
                req_id="req",
                prompt_token_ids=(1, 2, 3, 4, 5, 6),
                sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
                block_ids=((1, 2, 3), (4, 5, 6)),
                num_computed_tokens=0,
            ),
        ),
        scheduled_cached_reqs=CachedRequestData.empty(),
        num_scheduled_tokens={"req": 6},
        total_num_scheduled_tokens=6,
        sparse_plan=SparseKVPlan(
            (
                LayerSparsePlan(
                    request_id="req",
                    layer_idx=0,
                    selected_block_indices=(0, 2),
                ),
            )
        ),
    )

    runner._update_states(scheduler_output)
    runner._materialize_read_block_table(scheduler_output)
    input_batch = runner._require_input_batch()

    assert input_batch.block_table[0].get_cpu_tensor()[0, :2].tolist() == [1, 3]
    assert input_batch.block_table[0].block_indices.cpu[0, :2].tolist() == [0, 2]
    assert int(input_batch.block_table[0].num_blocks_per_row[0]) == 2
    assert input_batch.block_table[1].get_cpu_tensor()[0, :3].tolist() == [4, 5, 6]
    assert input_batch.block_table[1].block_indices.cpu[0, :3].tolist() == [0, 1, 2]


def test_model_runner_prepares_inputs_from_input_batch_buffers() -> None:
    runner = make_runner_with_tiny_model()
    kv_config = make_kv_config()
    runner.initialize_kv_cache(kv_config)

    scheduler_output = make_manual_scheduler_output()
    runner._update_states(scheduler_output)
    input_batch = runner._require_input_batch()
    assert input_batch.token_ids_cpu[0, :2].tolist() == [7, 8]
    assert input_batch.num_tokens[0] == 2
    assert input_batch.num_prompt_tokens[0] == 2
    assert input_batch.num_computed_tokens_cpu[0] == 0

    (
        logits_indices,
        sampled_request_ids,
        sampling_params,
        num_prefill_reqs,
        num_decode_reqs,
        max_query_len,
        max_seq_len,
    ) = runner._prepare_inputs(scheduler_output, (2,))
    attn_metadata = runner._build_attention_metadata(
        scheduler_output=scheduler_output,
        num_reqs=1,
        num_scheduled_tokens=2,
        num_prefill_reqs=num_prefill_reqs,
        num_decode_reqs=num_decode_reqs,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
    )

    assert runner.input_ids is not None
    assert runner.positions is not None
    assert runner.input_ids[:2].tolist() == [7, 8]
    assert runner.positions[:2].tolist() == [0, 1]
    assert logits_indices.tolist() == [1]
    assert sampled_request_ids == ("req",)
    assert len(sampling_params) == 1
    assert attn_metadata.kv_cache_tensor is runner.kv_cache_tensor


def test_model_runner_batch_tracks_chunked_prefill_and_decode_tokens() -> None:
    runner = make_runner_with_tiny_model()
    kv_config = make_kv_config()
    runner.initialize_kv_cache(kv_config)
    scheduler = Scheduler(
        KVCoreConfig(
            model_config=ModelConfig(model="unused", attn_backend="torch_paged"),
            scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_scheduled_tokens=4),
            device_config=DeviceConfig(device="cpu"),
        ),
        kv_config,
    )
    scheduler.add_request(make_request(token_ids=[1, 2, 3, 4, 5, 6], max_tokens=2))

    first = scheduler.schedule()
    runner._update_states(first)
    first_tokens = tuple(
        first.num_scheduled_tokens[req_id]
        for req_id in runner._require_input_batch().req_ids
    )
    first_logits_indices, first_sampled_request_ids, *_ = runner._prepare_inputs(
        first,
        first_tokens,
    )
    assert runner.input_ids is not None
    assert runner.positions is not None
    assert runner.input_ids[:4].tolist() == [1, 2, 3, 4]
    assert runner.positions[:4].tolist() == [0, 1, 2, 3]
    assert first_logits_indices.tolist() == []
    assert first_sampled_request_ids == ()
    scheduler.update_from_outputs(first, {})

    second = scheduler.schedule()
    runner._update_states(second)
    second_tokens = tuple(
        second.num_scheduled_tokens[req_id]
        for req_id in runner._require_input_batch().req_ids
    )
    second_logits_indices, second_sampled_request_ids, *_ = runner._prepare_inputs(
        second,
        second_tokens,
    )
    assert runner.input_ids[:2].tolist() == [5, 6]
    assert runner.positions[:2].tolist() == [4, 5]
    assert second_logits_indices.tolist() == [1]
    assert second_sampled_request_ids == ("req",)
    runner._require_input_batch().record_sampled_tokens(("req",), (9,))
    scheduler.update_from_outputs(second, {"req": 9})

    third = scheduler.schedule()
    runner._update_states(third)
    third_tokens = tuple(
        third.num_scheduled_tokens[req_id]
        for req_id in runner._require_input_batch().req_ids
    )
    third_logits_indices, third_sampled_request_ids, *_ = runner._prepare_inputs(
        third,
        third_tokens,
    )
    assert runner.input_ids[:1].tolist() == [9]
    assert runner.positions[:1].tolist() == [6]
    assert third_logits_indices.tolist() == [0]
    assert third_sampled_request_ids == ("req",)


def test_model_runner_execute_model_uses_forward_context_not_model_kwargs() -> None:
    runner = ModelRunner(make_runner_config())
    spy_model = SpyCausalLM()
    runner.model = spy_model
    kv_config = make_kv_config(num_layers=1)
    runner.initialize_kv_cache(kv_config)
    assert runner.kv_cache_tensor is not None
    runner.kv_cache_tensor.fill_(1)

    model_runner_output = runner.execute_model(make_manual_scheduler_output())

    assert len(spy_model.calls) == 1
    assert spy_model.calls[0]["input_ids"].tolist() == [7, 8]
    assert spy_model.calls[0]["positions"].tolist() == [0, 1]
    assert model_runner_output is None
    assert runner.execute_model_state is not None
    model_runner_output = runner.sample_tokens()
    assert model_runner_output.req_ids == ["req"]
    assert model_runner_output.req_id_to_index == {"req": 0}
    assert model_runner_output.sampled_token_ids == [[5]]
    assert runner.execute_model_state is None
    assert runner.last_step_stats is not None
    assert runner.last_step_stats.num_reqs == 1
    assert runner.last_step_stats.num_scheduled_tokens == 2
    assert runner.last_step_stats.num_zeroed_blocks == 1
    assert torch.count_nonzero(runner.kv_cache_tensor[:, 1]).item() == 0


def test_model_runner_execute_model_samples_requested_positions() -> None:
    runner = make_runner_with_tiny_model()
    kv_config = make_kv_config()
    runner.initialize_kv_cache(kv_config)
    scheduler = Scheduler(
        KVCoreConfig(
            model_config=ModelConfig(model="unused", attn_backend="torch_paged"),
            scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_scheduled_tokens=8),
            device_config=DeviceConfig(device="cpu"),
        ),
        kv_config,
    )

    scheduler.add_request(make_request(token_ids=[1, 2, 3, 4]))
    scheduler_output = scheduler.schedule()
    model_runner_output = runner.execute_model(scheduler_output)
    assert model_runner_output is None
    model_runner_output = runner.sample_tokens()

    assert model_runner_output.req_ids == ["req"]
    assert len(model_runner_output.sampled_token_ids) == 1
    assert len(model_runner_output.sampled_token_ids[0]) == 1


def test_model_runner_returns_block_score_updates_when_sparse_enabled() -> None:
    runner = ModelRunner(
        KVCoreConfig(
            model_config=ModelConfig(model="unused", attn_backend="torch_paged"),
            sparse_kv_config=SparseKVConfig(mode="dynamic"),
            device_config=DeviceConfig(device="cpu"),
        )
    )
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
            model_config=ModelConfig(
                model="unused",
                attn_backend="torch_paged",
                hf_config=hf_config,
            ),
            sparse_kv_config=SparseKVConfig(mode="dynamic"),
            device_config=DeviceConfig(device="cpu"),
        )
    )
    kv_config = make_kv_config()
    runner.initialize_kv_cache(kv_config)
    scheduler = Scheduler(
        KVCoreConfig(
            model_config=ModelConfig(model="unused", attn_backend="torch_paged"),
            scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_scheduled_tokens=8),
            sparse_kv_config=SparseKVConfig(mode="dynamic"),
            device_config=DeviceConfig(device="cpu"),
        ),
        kv_config,
    )
    scheduler.add_request(make_request(token_ids=[1, 2, 3, 4]))

    scheduler_output = scheduler.schedule()
    assert runner.execute_model(scheduler_output) is None
    model_runner_output = runner.sample_tokens()

    assert model_runner_output.block_score_updates
    assert {update.layer_idx for update in model_runner_output.block_score_updates} == {0, 1}
    assert model_runner_output.block_score_updates[0].request_id == "req"
