from __future__ import annotations

import torch

from kvcore.config import KVCoreConfig, ModelConfig
from kvcore.kv.compression import KVCompressionConfig, RandomKVBlockCompressor
from kvcore.kv.kv_manager import KVManager, KVManagerConfig
from kvcore.kv.single_type_kv_manager import KVLayerSpec
from kvcore.sched.scheduler import Scheduler
from kvcore.utils.request import Request
from kvcore.utils.sampling_params import SamplingParams


def make_kv_config() -> KVManagerConfig:
    return KVManagerConfig(
        num_gpu_blocks=16,
        max_model_len=16,
        layer_specs=(
            KVLayerSpec(0, 2, 2, 8, torch.float16),
            KVLayerSpec(1, 2, 2, 8, torch.float16),
        ),
    )


def make_config() -> KVCoreConfig:
    return KVCoreConfig(model_config=ModelConfig(model="unused", attn_backend="torch_paged"))


def make_request(request_id: str = "req") -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3, 4, 5, 6],
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
    )


def test_random_kv_block_compressor_permanently_evicts_selected_blocks() -> None:
    kv_manager = KVManager(make_kv_config())
    request = make_request()
    kv_manager.allocate_slots(request, num_new_tokens=6)
    original_ids = kv_manager.get_block_ids("req")

    compressor = RandomKVBlockCompressor(
        KVCompressionConfig(drop_ratio=1.0, max_blocks=1, seed=0)
    )
    result = compressor.compress(kv_manager, ("req",))
    compressed_ids = kv_manager.get_block_ids("req")

    assert result.num_evicted_blocks == 2
    assert result.selections[0].request_id == "req"
    assert compressed_ids[0].count(0) == 1
    assert compressed_ids[1].count(0) == 1
    assert compressed_ids[0][-1] == original_ids[0][-1]
    assert compressed_ids[1][-1] == original_ids[1][-1]


def test_scheduler_exposes_random_kv_compression_entrypoint() -> None:
    scheduler = Scheduler(make_config(), make_kv_config())
    request = make_request()
    scheduler.add_request(request)
    scheduler.schedule()

    result = scheduler.compress_kv_cache("req", drop_ratio=1.0, seed=0, max_blocks=1)

    assert result.num_evicted_blocks == 2
    assert scheduler.kv_manager.get_block_ids("req")[0].count(0) == 1
