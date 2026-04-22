from __future__ import annotations

import pytest

from kvcore.kv.block_pool import BlockPool
from kvcore.kv.kv_metrics import KVCacheMetricsCollector
from kvcore.kv.kv_utils import (
    BlockHash,
    FreeKVBlockQueue,
    KVBlock,
    get_group_id,
    get_request_block_hasher,
    make_block_hash_with_group_id,
)
from kvcore.utils.request import Request
from kvcore.utils.sampling_params import SamplingParams


def test_block_hash_with_group_id_roundtrip() -> None:
    block_hash = BlockHash(b"abc")
    key = make_block_hash_with_group_id(block_hash, 7)

    assert key.startswith(block_hash)
    assert get_group_id(key) == 7


def test_kv_block_hash_can_only_be_set_once() -> None:
    block = KVBlock(block_id=1)
    key = make_block_hash_with_group_id(BlockHash(b"abc"), 0)
    block.block_hash = key

    with pytest.raises(ValueError, match="already has a hash"):
        block.block_hash = key

    block.reset_hash()
    assert block.block_hash is None


def test_free_queue_pop_remove_and_append_order() -> None:
    blocks = [KVBlock(block_id=i) for i in range(4)]
    queue = FreeKVBlockQueue(blocks)

    assert [block.block_id for block in queue.get_all_free_blocks()] == [0, 1, 2, 3]

    first = queue.popleft()
    assert first.block_id == 0
    assert queue.num_free_blocks == 3

    queue.remove(blocks[2])
    assert [block.block_id for block in queue.get_all_free_blocks()] == [1, 3]

    queue.append(first)
    assert [block.block_id for block in queue.get_all_free_blocks()] == [1, 3, 0]


def test_block_pool_reserves_null_block() -> None:
    pool = BlockPool(num_gpu_blocks=4)

    assert pool.null_block.block_id == 0
    assert pool.null_block.is_null
    assert pool.get_num_free_blocks() == 3
    assert [block.block_id for block in pool.free_block_queue.get_all_free_blocks()] == [1, 2, 3]


def test_block_pool_allocates_and_frees_blocks() -> None:
    pool = BlockPool(num_gpu_blocks=4)

    blocks = pool.get_new_blocks(2)
    assert [block.block_id for block in blocks] == [1, 2]
    assert [block.ref_cnt for block in blocks] == [1, 1]
    assert pool.get_num_free_blocks() == 1

    pool.free_blocks(blocks)
    assert [block.ref_cnt for block in blocks] == [0, 0]
    assert pool.get_num_free_blocks() == 3


def test_block_pool_touch_removes_free_cached_block() -> None:
    pool = BlockPool(num_gpu_blocks=3)
    block = pool.get_new_blocks(1)[0]
    pool.free_blocks([block])

    assert block.ref_cnt == 0
    pool.touch([block])

    assert block.ref_cnt == 1
    assert block not in pool.free_block_queue.get_all_free_blocks()


def test_block_pool_caches_and_gets_full_blocks() -> None:
    block_size = 2
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=1),
        block_hasher=get_request_block_hasher(block_size),
    )
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True)
    blocks = pool.get_new_blocks(2)

    pool.cache_full_blocks(
        request,
        blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        kv_cache_group_id=0,
    )

    cached = pool.get_cached_block(request.block_hashes[0], [0])
    assert cached == [blocks[0]]
    assert blocks[0].block_hash is not None
    assert blocks[1].block_hash is not None


def test_block_pool_evicts_cached_block_when_reallocated() -> None:
    block_size = 2
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=1),
        block_hasher=get_request_block_hasher(block_size),
    )
    pool = BlockPool(num_gpu_blocks=2, enable_caching=True)
    block = pool.get_new_blocks(1)[0]
    pool.cache_full_blocks(
        request,
        [block],
        num_cached_blocks=0,
        num_full_blocks=1,
        kv_cache_group_id=0,
    )
    pool.free_blocks([block])

    reallocated = pool.get_new_blocks(1)[0]

    assert reallocated is block
    assert reallocated.block_hash is None
    assert pool.get_cached_block(request.block_hashes[0], [0]) is None


def test_block_pool_reset_prefix_cache_requires_all_blocks_free() -> None:
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True)
    allocated = pool.get_new_blocks(1)

    assert not pool.reset_prefix_cache()

    pool.free_blocks(allocated)
    assert pool.reset_prefix_cache()


def test_metrics_collector_records_access_and_eviction_event() -> None:
    block_size = 2
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=1),
        block_hasher=get_request_block_hasher(block_size),
    )
    metrics = KVCacheMetricsCollector(sample_rate=1.0)
    pool = BlockPool(num_gpu_blocks=2, enable_caching=True, metrics_collector=metrics)

    block = pool.get_new_blocks(1)[0]
    pool.touch([block])
    assert block.block_id in metrics.block_metrics
    assert len(metrics.block_metrics[block.block_id].access_history) == 1

    pool.cache_full_blocks(
        request,
        [block],
        num_cached_blocks=0,
        num_full_blocks=1,
        kv_cache_group_id=0,
    )
    pool.free_blocks([block])
    pool.free_blocks([block])
    pool.get_new_blocks(1)

    events = metrics.drain_events()
    assert len(events) == 1
    assert events[0].lifetime_seconds >= 0
    assert events[0].idle_seconds >= 0
    assert events[0].reuse_gaps_seconds == ()
    assert metrics.drain_events() == []


def test_metrics_collector_sample_rate_validation_and_reset() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        KVCacheMetricsCollector(sample_rate=0.0)

    metrics = KVCacheMetricsCollector(sample_rate=1.0)
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, metrics_collector=metrics)
    blocks = pool.get_new_blocks(1)

    assert metrics.block_metrics

    pool.free_blocks(blocks)
    assert pool.reset_prefix_cache()
    assert metrics.block_metrics == {}
    assert metrics.drain_events() == []
