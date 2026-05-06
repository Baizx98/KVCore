from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from kvcore.kv.block_pool import BlockPool
from kvcore.kv.kv_utils import KVBlock
from kvcore.kv.sparse import BlockSparseState
from kvcore.utils.request import Request


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


class KVCacheType(StrEnum):
    FULL = "full"
    SLIDING_WINDOW = "sliding_window"


@dataclass(frozen=True, slots=True)
class KVLayerSpec:
    layer_idx: int
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: object
    cache_type: KVCacheType = KVCacheType.FULL
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        if self.layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {self.layer_idx}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.head_size <= 0:
            raise ValueError(f"head_size must be positive, got {self.head_size}")
        if self.cache_type == KVCacheType.SLIDING_WINDOW and not self.sliding_window:
            raise ValueError("sliding_window must be set for sliding-window KV layers")


@dataclass(frozen=True, slots=True)
class LayerBlockSelection:
    request_id: str
    layer_idx: int
    block_indices: set[int]

    def __post_init__(self) -> None:
        if self.layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {self.layer_idx}")
        if any(index < 0 for index in self.block_indices):
            raise ValueError("block_indices must be non-negative logical block indices")


@dataclass(frozen=True, slots=True)
class LayerEvictionResult:
    request_id: str
    layer_idx: int
    evicted_block_indices: tuple[int, ...] = ()
    evicted_block_ids: tuple[int, ...] = ()
    skipped_block_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class EvictionResult:
    layer_results: tuple[LayerEvictionResult, ...]

    @property
    def evicted_block_ids(self) -> tuple[int, ...]:
        return tuple(
            block_id
            for layer_result in self.layer_results
            for block_id in layer_result.evicted_block_ids
        )


class SingleTypeKVManager(ABC):
    def __init__(
        self,
        kv_cache_spec: KVLayerSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
    ) -> None:
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self.kv_cache_group_id = kv_cache_group_id
        self.new_block_ids: list[int] = []
        self.block_size = self.kv_cache_spec.block_size
        self.req_to_blocks: defaultdict[str, list[KVBlock]] = defaultdict(list)
        self.num_cached_blocks: dict[str, int] = {}
        self.permanently_evicted_blocks: defaultdict[str, set[int]] = defaultdict(set)
        self.block_sparse_states: defaultdict[str, dict[int, BlockSparseState]] = (
            defaultdict(dict)
        )
        self._null_block = self.block_pool.null_block

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVBlock] = (),
    ) -> int:
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))
        num_new_blocks = max(num_required_blocks - num_req_blocks - len(new_computed_blocks), 0)
        num_evictable_hits = sum(
            block.ref_cnt == 0 and not block.is_null for block in new_computed_blocks
        )
        return num_new_blocks + num_evictable_hits

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVBlock],
    ) -> None:
        if not new_computed_blocks:
            return
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        self.req_to_blocks[request_id].extend(new_computed_blocks)
        self.num_cached_blocks[request_id] = len(self.req_to_blocks[request_id])
        self._sync_sparse_states(request_id)

    def allocate_new_blocks(self, request_id: str, num_tokens: int) -> list[KVBlock]:
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        self.new_block_ids.extend(block.block_id for block in new_blocks)
        self._sync_sparse_states(request_id)
        return new_blocks

    def take_new_block_ids(self) -> list[int]:
        block_ids = self.new_block_ids
        self.new_block_ids = []
        return block_ids

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        if not self.enable_caching:
            return
        num_cached_blocks = self.num_cached_blocks.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size
        num_full_blocks = min(num_full_blocks, len(request.block_hashes))
        if num_cached_blocks >= num_full_blocks:
            return

        blocks = self.req_to_blocks[request.request_id]
        evicted_indices = self.permanently_evicted_blocks.get(request.request_id, set())
        cache_start = num_cached_blocks
        while cache_start < num_full_blocks:
            while cache_start < num_full_blocks and cache_start in evicted_indices:
                cache_start += 1
            cache_end = cache_start
            while cache_end < num_full_blocks and cache_end not in evicted_indices:
                cache_end += 1
            if cache_start < cache_end:
                self.block_pool.cache_full_blocks(
                    request=request,
                    blocks=blocks,
                    num_cached_blocks=cache_start,
                    num_full_blocks=cache_end,
                    kv_cache_group_id=self.kv_cache_group_id,
                )
            cache_start = cache_end
        self.num_cached_blocks[request.request_id] = max(num_cached_blocks, num_full_blocks)

    def free(self, request_id: str) -> None:
        req_blocks = self.req_to_blocks.pop(request_id, [])
        self.block_pool.free_blocks(reversed(req_blocks))
        self.num_cached_blocks.pop(request_id, None)
        self.permanently_evicted_blocks.pop(request_id, None)
        self.block_sparse_states.pop(request_id, None)

    def evict_blocks(self, request_id: str, block_indices: set[int]) -> LayerEvictionResult:
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return LayerEvictionResult(
                request_id=request_id,
                layer_idx=self.kv_cache_spec.layer_idx,
                skipped_block_indices=tuple(sorted(block_indices)),
            )

        evicted_indices: list[int] = []
        evicted_block_ids: list[int] = []
        skipped_indices: list[int] = []
        blocks_to_free: list[KVBlock] = []
        permanent_set = self.permanently_evicted_blocks[request_id]

        for block_index in sorted(block_indices):
            if block_index >= len(blocks) or block_index in permanent_set:
                skipped_indices.append(block_index)
                continue
            block = blocks[block_index]
            if block.is_null:
                permanent_set.add(block_index)
                self._mark_permanently_evicted(request_id, block_index)
                skipped_indices.append(block_index)
                continue

            blocks[block_index] = self._null_block
            permanent_set.add(block_index)
            self._mark_permanently_evicted(request_id, block_index)
            evicted_indices.append(block_index)
            evicted_block_ids.append(block.block_id)
            blocks_to_free.append(block)

        if blocks_to_free:
            # Free in reverse logical order so tail blocks stay first in eviction order.
            self.block_pool.free_blocks(reversed(blocks_to_free))

        return LayerEvictionResult(
            request_id=request_id,
            layer_idx=self.kv_cache_spec.layer_idx,
            evicted_block_indices=tuple(evicted_indices),
            evicted_block_ids=tuple(evicted_block_ids),
            skipped_block_indices=tuple(skipped_indices),
        )

    def get_sparse_states(self, request_id: str) -> dict[int, BlockSparseState]:
        self._sync_sparse_states(request_id)
        return self.block_sparse_states.get(request_id, {})

    def update_block_scores(
        self,
        request_id: str,
        block_scores: dict[int, float],
        *,
        step_id: int,
        ema_alpha: float,
    ) -> None:
        self._sync_sparse_states(request_id)
        states = self.block_sparse_states.get(request_id, {})
        for block_index, score in block_scores.items():
            state = states.get(block_index)
            if state is None or state.is_permanently_evicted:
                continue
            previous = state.ema_score
            state.score = score
            state.ema_score = (
                score if previous is None else ema_alpha * previous + (1.0 - ema_alpha) * score
            )
            state.last_scored_step = step_id

    def mark_dynamic_selection(
        self,
        request_id: str,
        keep_indices: set[int],
        *,
        step_id: int,
    ) -> None:
        self._sync_sparse_states(request_id)
        for block_index, state in self.block_sparse_states.get(request_id, {}).items():
            if state.is_permanently_evicted:
                continue
            state.was_dynamic_skipped = block_index not in keep_indices
            state.last_selected_step = step_id

    def _sync_sparse_states(self, request_id: str) -> None:
        blocks = self.req_to_blocks.get(request_id, [])
        states = self.block_sparse_states[request_id]
        evicted = self.permanently_evicted_blocks.get(request_id, set())
        for block_index, block in enumerate(blocks):
            state = states.get(block_index)
            if state is None:
                states[block_index] = BlockSparseState(
                    request_id=request_id,
                    layer_idx=self.kv_cache_spec.layer_idx,
                    logical_block_idx=block_index,
                    physical_block_id=block.block_id,
                    is_permanently_evicted=block_index in evicted or block.is_null,
                )
                continue
            state.physical_block_id = block.block_id
            state.is_permanently_evicted = block_index in evicted or block.is_null
        for block_index in tuple(states):
            if block_index >= len(blocks):
                states.pop(block_index, None)

    def _mark_permanently_evicted(self, request_id: str, block_index: int) -> None:
        self._sync_sparse_states(request_id)
        state = self.block_sparse_states[request_id].get(block_index)
        if state is None:
            return
        state.is_permanently_evicted = True
        state.physical_block_id = self._null_block.block_id

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        blocks = self.req_to_blocks.get(running_request_id, [])
        num_common_blocks = 0
        for block_index, block in enumerate(blocks):
            if block_index in self.permanently_evicted_blocks.get(running_request_id, set()):
                break
            if block.ref_cnt == len(self.req_to_blocks):
                num_common_blocks += 1
            else:
                break
        return num_common_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return 0

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: Sequence[object],
        max_length: int,
        kv_cache_group_id: int,
        block_pool: BlockPool,
        kv_cache_spec: KVLayerSpec,
    ) -> list[KVBlock]:
        raise NotImplementedError


class FullAttentionKVManager(SingleTypeKVManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: Sequence[object],
        max_length: int,
        kv_cache_group_id: int,
        block_pool: BlockPool,
        kv_cache_spec: KVLayerSpec,
    ) -> list[KVBlock]:
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks: list[KVBlock] = []
        for block_hash in block_hashes[:max_num_blocks]:
            cached_blocks = block_pool.get_cached_block(block_hash, [kv_cache_group_id])
            if cached_blocks is None:
                break
            computed_blocks.append(cached_blocks[0])
        return computed_blocks


class SlidingWindowKVManager(SingleTypeKVManager):
    def __init__(
        self,
        kv_cache_spec: KVLayerSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
    ) -> None:
        super().__init__(
            kv_cache_spec=kv_cache_spec,
            block_pool=block_pool,
            enable_caching=enable_caching,
            kv_cache_group_id=kv_cache_group_id,
        )
        if self.kv_cache_spec.sliding_window is None:
            raise ValueError("sliding_window must be set for SlidingWindowKVManager")
        self.sliding_window = self.kv_cache_spec.sliding_window

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return max(0, num_computed_tokens - self.sliding_window + 1)

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: Sequence[object],
        max_length: int,
        kv_cache_group_id: int,
        block_pool: BlockPool,
        kv_cache_spec: KVLayerSpec,
    ) -> list[KVBlock]:
        raise NotImplementedError("Sliding-window cache hits are reserved for later work")


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVLayerSpec,
    block_pool: BlockPool,
    enable_caching: bool,
    kv_cache_group_id: int,
) -> SingleTypeKVManager:
    manager_cls: type[SingleTypeKVManager]
    if kv_cache_spec.cache_type == KVCacheType.FULL:
        manager_cls = FullAttentionKVManager
    elif kv_cache_spec.cache_type == KVCacheType.SLIDING_WINDOW:
        manager_cls = SlidingWindowKVManager
    else:
        raise ValueError(f"Unsupported KV cache type: {kv_cache_spec.cache_type}")
    return manager_cls(
        kv_cache_spec=kv_cache_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=kv_cache_group_id,
    )


__all__ = [
    "EvictionResult",
    "FullAttentionKVManager",
    "KVCacheType",
    "KVLayerSpec",
    "LayerBlockSelection",
    "LayerEvictionResult",
    "SingleTypeKVManager",
    "SlidingWindowKVManager",
    "get_manager_for_kv_cache_spec",
]
