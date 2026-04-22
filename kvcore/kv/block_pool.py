from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from kvcore.kv.kv_metrics import KVCacheMetricsCollector
from kvcore.kv.kv_utils import (
    BlockHash,
    BlockHashWithGroupId,
    FreeKVBlockQueue,
    KVBlock,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
)
from kvcore.utils.request import Request


class BlockHashToBlockMap:
    def __init__(self) -> None:
        self._cache: dict[BlockHashWithGroupId, KVBlock | dict[int, KVBlock]] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVBlock | None:
        blocks = self._cache.get(key)
        if blocks is None:
            return None
        if isinstance(blocks, KVBlock):
            return blocks
        if isinstance(blocks, dict):
            return next(iter(blocks.values()), None)
        self._unexpected_blocks_type(blocks)

    def insert(self, key: BlockHashWithGroupId, block: KVBlock) -> None:
        blocks = self._cache.get(key)
        if blocks is None:
            self._cache[key] = block
        elif isinstance(blocks, KVBlock):
            self._cache[key] = {
                blocks.block_id: blocks,
                block.block_id: block,
            }
        elif isinstance(blocks, dict):
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVBlock | None:
        blocks = self._cache.pop(key, None)
        if blocks is None:
            return None
        if isinstance(blocks, KVBlock):
            if blocks.block_id == block_id:
                return blocks
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            block = blocks.pop(block_id, None)
            if blocks:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)

    def __len__(self) -> int:
        return len(self._cache)

    @staticmethod
    def _unexpected_blocks_type(blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    def __init__(
        self,
        num_gpu_blocks: int,
        *,
        enable_caching: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        if num_gpu_blocks <= 0:
            raise ValueError(f"num_gpu_blocks must be positive, got {num_gpu_blocks}")

        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.metrics_collector = metrics_collector
        self.blocks = [KVBlock(block_id=i) for i in range(num_gpu_blocks)]
        self.free_block_queue = FreeKVBlockQueue(self.blocks)
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

    def get_new_blocks(self, num_blocks: int) -> list[KVBlock]:
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        blocks = self.free_block_queue.popleft_n(num_blocks)
        for block in blocks:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            if block.ref_cnt != 0:
                raise RuntimeError(f"Allocated block has non-zero ref_cnt: {block}")
            block.ref_cnt = 1
            if self.metrics_collector is not None:
                self.metrics_collector.on_block_allocated(block)
        return blocks

    def touch(self, blocks: Sequence[KVBlock]) -> None:
        for block in blocks:
            if block.is_null:
                continue
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector is not None:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVBlock]) -> None:
        freeable_blocks: list[KVBlock] = []
        for block in ordered_blocks:
            if block.is_null:
                continue
            if block.ref_cnt <= 0:
                raise RuntimeError(f"Cannot free block with ref_cnt={block.ref_cnt}: {block}")
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                freeable_blocks.append(block)
        self.free_block_queue.append_n(freeable_blocks)

    def get_cached_block(
        self,
        block_hash: BlockHash,
        kv_cache_group_ids: list[int],
    ) -> list[KVBlock] | None:
        cached_blocks: list[KVBlock] = []
        for group_id in kv_cache_group_ids:
            key = make_block_hash_with_group_id(block_hash, group_id)
            block = self.cached_block_hash_to_block.get_one_block(key)
            if block is None:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVBlock],
        *,
        num_cached_blocks: int,
        num_full_blocks: int,
        kv_cache_group_id: int = 0,
    ) -> None:
        if not self.enable_caching:
            return
        if num_cached_blocks >= num_full_blocks:
            return
        if len(request.block_hashes) < num_full_blocks:
            raise ValueError(
                f"Request has {len(request.block_hashes)} block hashes, "
                f"but {num_full_blocks} full blocks need caching"
            )

        for index in range(num_cached_blocks, num_full_blocks):
            block = blocks[index]
            if block.is_null:
                continue
            if block.block_hash is not None:
                continue
            key = make_block_hash_with_group_id(
                request.block_hashes[index],
                kv_cache_group_id,
            )
            block.block_hash = key
            self.cached_block_hash_to_block.insert(key, block)

    def reset_prefix_cache(self) -> bool:
        if self.num_gpu_blocks - self.get_num_free_blocks() != 1:
            return False
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        for block in self.blocks:
            block.reset_hash()
        if self.metrics_collector is not None:
            self.metrics_collector.reset()
        return True

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        total_usable_blocks = self.num_gpu_blocks - 1
        if total_usable_blocks <= 0:
            return 0.0
        return 1.0 - (self.get_num_free_blocks() / total_usable_blocks)

    def _maybe_evict_cached_block(self, block: KVBlock) -> bool:
        if self.metrics_collector is not None:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            return False
        evicted = self.cached_block_hash_to_block.pop(block_hash, block.block_id)
        if evicted is None:
            return False
        block.reset_hash()
        return True


__all__ = [
    "BlockHashToBlockMap",
    "BlockPool",
    "get_block_hash",
    "get_group_id",
]
