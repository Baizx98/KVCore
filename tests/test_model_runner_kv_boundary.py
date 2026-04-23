from __future__ import annotations

import torch
from transformers import LlamaConfig

from kvcore.kv.kv_manager import KVManagerConfig
from kvcore.kv.single_type_kv_manager import KVLayerSpec, LayerBlockSelection
from kvcore.model.kv_runtime import SparseComputePlan
from kvcore.model.model_loader import ModelLoadConfig
from kvcore.model.model_runner import ModelRunner
from kvcore.model.models.llama3 import Llama3ForCausalLM
from kvcore.sched.scheduler import Scheduler
from kvcore.utils.request import Request
from kvcore.utils.sampling_params import SamplingParams


def make_runner_with_tiny_model() -> ModelRunner:
    runner = ModelRunner(ModelLoadConfig(model="unused", device="cpu"))
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
    runner.model = Llama3ForCausalLM(config)
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


def make_request() -> Request:
    return Request(
        request_id="req",
        prompt_token_ids=[1, 2, 3, 4, 5, 6],
        sampling_params=SamplingParams(max_tokens=1),
    )


def test_model_runner_initializes_global_kv_cache_tensor() -> None:
    runner = make_runner_with_tiny_model()

    kv_cache_tensor = runner.initialize_kv_cache(make_kv_config())

    assert runner.kv_cache_tensor is not None
    assert kv_cache_tensor is runner.kv_cache_tensor
    assert runner.kv_cache_tensor.shape == (2, 16, 2, 2, 8)
    assert not hasattr(runner, "kv_manager")
    for layer in runner.model.model.layers:
        assert layer.self_attn.attn.kv_cache is None


def test_model_runner_reads_scheduler_owned_kv_manager_for_metadata() -> None:
    runner = make_runner_with_tiny_model()
    kv_config = make_kv_config()
    runner.initialize_kv_cache(kv_config)
    scheduler = Scheduler(kv_config)
    kv_manager = scheduler.kv_manager

    request = make_request()
    assert kv_manager.allocate_slots(request, num_new_tokens=6) is not None
    original_block_ids = kv_manager.get_block_ids("req")

    sparse_plan = SparseComputePlan.from_selections(
        [LayerBlockSelection(request_id="req", layer_idx=0, block_indices={0, 2})]
    )
    metadata = runner.build_attention_metadata(
        kv_manager,
        ["req"],
        sparse_plan=sparse_plan,
    )

    assert metadata.block_tables[0].get_numpy_array()[0, :1].tolist() == [
        original_block_ids[0][1]
    ]
    assert metadata.skipped_block_indices[0] == ((0, 2),)
    assert metadata.block_tables[1].get_numpy_array()[0, :3].tolist() == original_block_ids[1]
    assert metadata.kv_cache_tensor is runner.kv_cache_tensor
    assert kv_manager.get_block_ids("req") == original_block_ids

    dense_metadata = runner.build_attention_metadata(kv_manager, ["req"])
    assert dense_metadata.block_tables[0].get_numpy_array()[0, :3].tolist() == original_block_ids[0]
