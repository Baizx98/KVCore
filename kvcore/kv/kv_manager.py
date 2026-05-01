from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from kvcore.kv.block_pool import BlockPool
from kvcore.kv.kv_metrics import KVCacheMetricsCollector
from kvcore.kv.kv_utils import KVBlock
from kvcore.kv.single_type_kv_manager import (
    EvictionResult,
    KVLayerSpec,
    LayerBlockSelection,
    SingleTypeKVManager,
    get_manager_for_kv_cache_spec,
)
from kvcore.utils.log import get_logger
from kvcore.utils.request import Request

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class KVManagerConfig:
    num_gpu_blocks: int
    max_model_len: int
    layer_specs: tuple[KVLayerSpec, ...]
    enable_caching: bool = True

    def __post_init__(self) -> None:
        if self.num_gpu_blocks <= 0:
            raise ValueError(
                f"num_gpu_blocks must be positive, got {self.num_gpu_blocks}"
            )
        if self.max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {self.max_model_len}")
        if not self.layer_specs:
            raise ValueError("layer_specs must be non-empty")
        layer_indices = [spec.layer_idx for spec in self.layer_specs]
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError(f"layer_specs contain duplicate layer_idx values: {layer_indices}")
        if layer_indices != list(range(len(layer_indices))):
            raise ValueError("layer_specs must be ordered by contiguous layer_idx values")


@dataclass(frozen=True, slots=True)
class KVCacheBlocks:
    blocks: tuple[Sequence[KVBlock], ...]

    def __add__(self, other: KVCacheBlocks) -> KVCacheBlocks:
        if len(self.blocks) != len(other.blocks):
            raise ValueError("Cannot add KVCacheBlocks with different layer counts")
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(left_blocks, right_blocks))
                for left_blocks, right_blocks in zip(self.blocks, other.blocks, strict=True)
            )
        )

    def get_block_ids(self, *, allow_none: bool = False) -> tuple[list[int], ...] | None:
        if allow_none and all(len(layer_blocks) == 0 for layer_blocks in self.blocks):
            return None
        return tuple([block.block_id for block in layer_blocks] for layer_blocks in self.blocks)

    def get_layer_block_ids(self, layer_idx: int) -> list[int]:
        return [block.block_id for block in self.blocks[layer_idx]]

    def get_unhashed_block_ids_all_layers(self) -> list[list[int]]:
        return [
            [
                block.block_id
                for block in layer_blocks
                if block.block_hash is None and not block.is_null
            ]
            for layer_blocks in self.blocks
        ]

    def new_empty(self) -> KVCacheBlocks:
        return KVCacheBlocks(tuple(() for _ in self.blocks))


class KVManager:
    def __init__(
        self,
        config: KVManagerConfig,
        *,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.config = config
        self.max_model_len = config.max_model_len
        self.enable_caching = config.enable_caching
        self.block_pool = BlockPool(
            config.num_gpu_blocks,
            enable_caching=config.enable_caching,
            metrics_collector=metrics_collector,
        )
        self.layer_managers: tuple[SingleTypeKVManager, ...] = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=layer_spec,
                block_pool=self.block_pool,
                enable_caching=config.enable_caching,
                kv_cache_group_id=layer_spec.layer_idx,
            )
            for layer_spec in config.layer_specs
        )
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(len(self.layer_managers)))
        )
        logger.info(
            "KVManager initialized num_gpu_blocks=%d max_model_len=%d "
            "num_layers=%d enable_caching=%s",
            config.num_gpu_blocks,
            config.max_model_len,
            len(self.layer_managers),
            config.enable_caching,
        )

    @property
    def usage(self) -> float:
        return self.block_pool.get_usage()

    @property
    def num_layers(self) -> int:
        return len(self.layer_managers)

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        if not self.enable_caching or request.skip_reading_prefix_cache:
            logger.debug(
                "Prefix cache lookup skipped request_id=%s enable_caching=%s "
                "skip_reading_prefix_cache=%s",
                request.request_id,
                self.enable_caching,
                request.skip_reading_prefix_cache,
            )
            return self.empty_kv_cache_blocks, 0

        max_cache_hit_length = max(request.num_tokens - 1, 0)
        if max_cache_hit_length == 0:
            return self.empty_kv_cache_blocks, 0

        hit_blocks_by_layer: list[list[KVBlock]] = []
        hit_lengths: list[int] = []
        for layer_idx, manager in enumerate(self.layer_managers):
            layer_hits = manager.find_longest_cache_hit(
                block_hashes=request.block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_id=layer_idx,
                block_pool=self.block_pool,
                kv_cache_spec=manager.kv_cache_spec,
            )
            hit_blocks_by_layer.append(layer_hits)
            hit_lengths.append(len(layer_hits) * manager.block_size)

        if not hit_lengths:
            return self.empty_kv_cache_blocks, 0

        hit_length = min(hit_lengths)
        logger.debug(
            "Prefix cache lookup request_id=%s hit_length=%d",
            request.request_id,
            hit_length,
        )
        aligned_blocks_by_layer: list[list[KVBlock]] = []
        for manager, layer_hits in zip(self.layer_managers, hit_blocks_by_layer, strict=True):
            num_hit_blocks = hit_length // manager.block_size
            aligned_blocks_by_layer.append(layer_hits[:num_hit_blocks])

        return self.create_kv_cache_blocks(tuple(aligned_blocks_by_layer)), hit_length

    def can_fit(
        self,
        request: Request,
        num_new_tokens: int,
        new_computed_blocks: KVCacheBlocks | None = None,
    ) -> bool:
        needed_blocks = self._get_num_blocks_to_allocate(
            request,
            num_new_tokens,
            new_computed_blocks,
        )
        free_blocks = self.block_pool.get_num_free_blocks()
        can_fit = needed_blocks <= free_blocks
        logger.debug(
            "KV can_fit request_id=%s num_new_tokens=%d needed_blocks=%d "
            "free_blocks=%d can_fit=%s",
            request.request_id,
            num_new_tokens,
            needed_blocks,
            free_blocks,
            can_fit,
        )
        return can_fit

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        new_computed_blocks: KVCacheBlocks | None = None,
    ) -> KVCacheBlocks | None:
        if num_new_tokens <= 0:
            raise ValueError(f"num_new_tokens must be positive, got {num_new_tokens}")
        if not self.can_fit(request, num_new_tokens, new_computed_blocks):
            logger.debug(
                "KV allocation rejected request_id=%s num_new_tokens=%d",
                request.request_id,
                num_new_tokens,
            )
            return None

        new_computed_block_groups = self._normalize_new_computed_blocks(new_computed_blocks)
        num_new_computed_tokens = self._get_num_new_computed_tokens(new_computed_block_groups)
        for manager, computed_blocks in zip(
            self.layer_managers,
            new_computed_block_groups,
            strict=True,
        ):
            manager.allocate_new_computed_blocks(request.request_id, computed_blocks)

        num_tokens_need_slot = min(
            request.num_computed_tokens + num_new_computed_tokens + num_new_tokens,
            self.max_model_len,
        )
        new_blocks = tuple(
            manager.allocate_new_blocks(request.request_id, num_tokens_need_slot)
            for manager in self.layer_managers
        )

        allocated = self.create_kv_cache_blocks(new_blocks)
        logger.debug(
            "KV allocated request_id=%s num_new_tokens=%d num_tokens_need_slot=%d "
            "layer0_blocks=%s",
            request.request_id,
            num_new_tokens,
            num_tokens_need_slot,
            allocated.get_layer_block_ids(0) if allocated.blocks else (),
        )
        return allocated

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        if not self.enable_caching:
            return
        for manager in self.layer_managers:
            manager.cache_blocks(request, num_computed_tokens)

    def free(self, request: Request) -> None:
        for manager in self.layer_managers:
            manager.free(request.request_id)
        logger.debug("KV freed request_id=%s", request.request_id)

    def evict_request_blocks(
        self,
        selections: Sequence[LayerBlockSelection],
    ) -> EvictionResult:
        layer_results = []
        for selection in selections:
            manager = self.layer_managers[selection.layer_idx]
            layer_results.append(
                manager.evict_blocks(selection.request_id, set(selection.block_indices))
            )
        result = EvictionResult(tuple(layer_results))
        logger.info(
            "KV evicted blocks selections=%d evicted_blocks=%d",
            len(selections),
            len(result.evicted_block_ids),
        )
        return result

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        return self.create_kv_cache_blocks(
            tuple(manager.req_to_blocks.get(request_id, []) for manager in self.layer_managers)
        )

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        return self.get_blocks(request_id).get_block_ids()

    def take_new_block_ids(self) -> list[int]:
        block_ids: list[int] = []
        for manager in self.layer_managers:
            block_ids.extend(manager.take_new_block_ids())
        if block_ids:
            logger.debug("KV new block ids to zero count=%d", len(block_ids))
        return block_ids

    def create_kv_cache_blocks(
        self,
        blocks: tuple[Sequence[KVBlock], ...],
    ) -> KVCacheBlocks:
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def _get_num_blocks_to_allocate(
        self,
        request: Request,
        num_new_tokens: int,
        new_computed_blocks: KVCacheBlocks | None,
    ) -> int:
        new_computed_block_groups = self._normalize_new_computed_blocks(new_computed_blocks)
        num_new_computed_tokens = self._get_num_new_computed_tokens(new_computed_block_groups)
        num_tokens_need_slot = min(
            request.num_computed_tokens + num_new_computed_tokens + num_new_tokens,
            self.max_model_len,
        )
        return sum(
            manager.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=num_tokens_need_slot,
                new_computed_blocks=computed_blocks,
            )
            for manager, computed_blocks in zip(
                self.layer_managers,
                new_computed_block_groups,
                strict=True,
            )
        )

    def _normalize_new_computed_blocks(
        self,
        new_computed_blocks: KVCacheBlocks | None,
    ) -> tuple[Sequence[KVBlock], ...]:
        if new_computed_blocks is None:
            return self.empty_kv_cache_blocks.blocks
        if len(new_computed_blocks.blocks) != self.num_layers:
            raise ValueError(
                "new_computed_blocks must match the number of KV layers, "
                f"got {len(new_computed_blocks.blocks)} for {self.num_layers} layers"
            )
        return new_computed_blocks.blocks

    def _get_num_new_computed_tokens(
        self,
        new_computed_block_groups: tuple[Sequence[KVBlock], ...],
    ) -> int:
        if not new_computed_block_groups:
            return 0
        return min(
            len(blocks) * manager.block_size
            for manager, blocks in zip(
                self.layer_managers,
                new_computed_block_groups,
                strict=True,
            )
        )


__all__ = [
    "KVCacheBlocks",
    "KVManager",
    "KVManagerConfig",
    "LayerBlockSelection",
]
