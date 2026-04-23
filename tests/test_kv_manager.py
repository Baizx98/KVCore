from __future__ import annotations

import torch

from kvcore.kv.kv_manager import KVManager, KVManagerConfig
from kvcore.kv.single_type_kv_manager import KVLayerSpec, LayerBlockSelection
from kvcore.utils.request import Request
from kvcore.utils.sampling_params import SamplingParams


def make_request(request_id: str, token_ids: list[int]) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=1),
    )


def make_kv_manager(num_layers: int = 2, num_gpu_blocks: int = 16) -> KVManager:
    return KVManager(
        KVManagerConfig(
            num_gpu_blocks=num_gpu_blocks,
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
    )


def test_kv_manager_allocates_one_block_table_per_layer_with_shared_pool() -> None:
    manager = make_kv_manager(num_layers=2, num_gpu_blocks=8)
    request = make_request("req", [1, 2, 3, 4])

    allocated = manager.allocate_slots(request, num_new_tokens=4)

    assert allocated is not None
    assert allocated.get_block_ids() == ([1, 2], [3, 4])
    assert manager.get_block_ids("req") == ([1, 2], [3, 4])
    assert manager.take_new_block_ids() == [1, 2, 3, 4]
    assert manager.block_pool.get_num_free_blocks() == 3


def test_kv_manager_permanent_eviction_replaces_sparse_indices_with_null_block() -> None:
    manager = make_kv_manager(num_layers=2, num_gpu_blocks=12)
    request = make_request("req", [1, 2, 3, 4, 5, 6])
    assert manager.allocate_slots(request, num_new_tokens=6) is not None
    before_layer_0 = manager.get_block_ids("req")[0]

    result = manager.evict_request_blocks(
        [
            LayerBlockSelection(
                request_id="req",
                layer_idx=0,
                block_indices={0, 2, 99},
            )
        ]
    )

    after = manager.get_block_ids("req")
    assert result.layer_results[0].evicted_block_indices == (0, 2)
    assert result.layer_results[0].skipped_block_indices == (99,)
    assert after[0] == [0, before_layer_0[1], 0]
    assert after[1] == [4, 5, 6]

    repeated = manager.evict_request_blocks(
        [LayerBlockSelection(request_id="req", layer_idx=0, block_indices={0, 2})]
    )
    assert repeated.layer_results[0].evicted_block_indices == ()
    assert repeated.layer_results[0].skipped_block_indices == (0, 2)


def test_kv_manager_free_releases_blocks_after_permanent_eviction() -> None:
    manager = make_kv_manager(num_layers=1, num_gpu_blocks=6)
    request = make_request("req", [1, 2, 3, 4])
    assert manager.allocate_slots(request, num_new_tokens=4) is not None
    manager.evict_request_blocks(
        [LayerBlockSelection(request_id="req", layer_idx=0, block_indices={1})]
    )

    manager.free(request)

    assert manager.block_pool.get_num_free_blocks() == 5
