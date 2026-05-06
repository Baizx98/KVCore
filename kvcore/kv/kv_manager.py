from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

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
from kvcore.kv.sparse import (
    BlockScoreUpdate,
    LayerSparsePlan,
    SparseKVMode,
    SparseKVPlan,
    SparseKVSelectionInterval,
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
        self.step_id = 0
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

    def build_sparse_plan(
        self,
        request_id: str,
        *,
        context_len: int,
        num_scheduled_tokens: int,
        is_prefill: bool,
        sparse_config: object,
    ) -> SparseKVPlan:
        full_block_ids = tuple(tuple(ids) for ids in self.get_block_ids(request_id))
        if not self._should_apply_sparse(is_prefill, sparse_config):
            return SparseKVPlan.empty()

        layer_plans: list[LayerSparsePlan] = []
        total_selected_blocks = 0
        total_candidate_blocks = 0
        for manager, layer_block_ids in zip(
            self.layer_managers,
            full_block_ids,
            strict=True,
        ):
            keep_indices = self._select_selected_block_indices(
                manager=manager,
                request_id=request_id,
                block_ids=layer_block_ids,
                context_len=context_len,
                num_scheduled_tokens=num_scheduled_tokens,
                sparse_config=sparse_config,
            )
            valid_indices = {
                block_index
                for block_index, block_id in enumerate(layer_block_ids)
                if block_id != 0
            }
            manager.mark_dynamic_selection(
                request_id,
                keep_indices,
                step_id=self.step_id,
            )
            visible_indices = tuple(
                block_index
                for block_index in sorted(keep_indices)
                if block_index < len(layer_block_ids)
            )
            total_candidate_blocks += len(valid_indices)
            total_selected_blocks += len(visible_indices)
            if set(visible_indices) != valid_indices:
                layer_plans.append(
                    LayerSparsePlan(
                        request_id=request_id,
                        layer_idx=manager.kv_cache_spec.layer_idx,
                        selected_block_indices=visible_indices,
                        is_sparse=True,
                    )
                )

        total_full_blocks = sum(len(layer_ids) for layer_ids in full_block_ids)
        total_skipped_blocks = total_candidate_blocks - total_selected_blocks
        logger.info(
            "Sparse KV plan request_id=%s context_len=%d scheduled_tokens=%d "
            "prefill=%s full_blocks=%d selected_blocks=%d skipped_blocks=%d "
            "mode=%s compression_ratio=%.3f",
            request_id,
            context_len,
            num_scheduled_tokens,
            is_prefill,
            total_full_blocks,
            total_selected_blocks,
            total_skipped_blocks,
            getattr(sparse_config, "mode", None),
            getattr(sparse_config, "compression_ratio", 0.0),
        )

        return SparseKVPlan(tuple(layer_plans))

    def update_block_scores(
        self,
        updates: Sequence[BlockScoreUpdate],
        *,
        ema_alpha: float,
    ) -> None:
        score_summaries: dict[tuple[str, int, str], tuple[set[int], int]] = {}
        for update in updates:
            if update.layer_idx < 0 or update.layer_idx >= self.num_layers:
                continue
            scores = dict(zip(update.logical_block_indices, update.scores, strict=True))
            if scores:
                summary_key = (update.request_id, update.step_id, update.score_kind)
                layers, num_blocks = score_summaries.setdefault(summary_key, (set(), 0))
                layers.add(update.layer_idx)
                score_summaries[summary_key] = (layers, num_blocks + len(scores))
            self.layer_managers[update.layer_idx].update_block_scores(
                update.request_id,
                scores,
                step_id=update.step_id,
                ema_alpha=ema_alpha,
            )
        for (request_id, step_id, score_kind), (
            layers,
            num_blocks,
        ) in score_summaries.items():
            logger.info(
                "Sparse KV score update request_id=%s layers=%d blocks=%d "
                "score_kind=%s step_id=%d",
                request_id,
                len(layers),
                num_blocks,
                score_kind,
                step_id,
            )

    def advance_step(self) -> None:
        self.step_id += 1

    def evict_unselected_sparse_blocks(
        self,
        request_id: str,
        *,
        context_len: int,
        sparse_config: object,
    ) -> EvictionResult:
        selections: list[LayerBlockSelection] = []
        full_block_ids = tuple(tuple(ids) for ids in self.get_block_ids(request_id))
        for manager, layer_block_ids in zip(
            self.layer_managers,
            full_block_ids,
            strict=True,
        ):
            valid_indices = {
                block_index
                for block_index, block_id in enumerate(layer_block_ids)
                if block_id != 0
            }
            if not valid_indices:
                continue
            selected_indices = self._select_selected_block_indices(
                manager=manager,
                request_id=request_id,
                block_ids=layer_block_ids,
                context_len=context_len,
                num_scheduled_tokens=0,
                sparse_config=sparse_config,
            )
            evict_indices = valid_indices - selected_indices
            if evict_indices:
                selections.append(
                    LayerBlockSelection(
                        request_id=request_id,
                        layer_idx=manager.kv_cache_spec.layer_idx,
                        block_indices=evict_indices,
                    )
                )
        if not selections:
            return EvictionResult(())
        return self.evict_request_blocks(selections)

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

    def _should_apply_sparse(self, is_prefill: bool, sparse_config: object) -> bool:
        mode = getattr(sparse_config, "mode", SparseKVMode.DISABLED.value)
        if mode == SparseKVMode.DISABLED.value:
            return False
        if is_prefill and not getattr(sparse_config, "enable_prefill_sparsity", False):
            return False
        if not is_prefill and not getattr(sparse_config, "enable_decode_sparsity", True):
            return False
        interval = getattr(
            sparse_config,
            "selection_interval",
            SparseKVSelectionInterval.STEP.value,
        )
        if interval == SparseKVSelectionInterval.STEP.value:
            return True
        if interval == SparseKVSelectionInterval.BLOCK.value:
            block_size = self.layer_managers[0].block_size
            return self.step_id % block_size == 0
        if interval == SparseKVSelectionInterval.N_TOKENS.value:
            interval_tokens = getattr(sparse_config, "selection_interval_tokens", None)
            return bool(interval_tokens and self.step_id % interval_tokens == 0)
        return False

    def _select_selected_block_indices(
        self,
        *,
        manager: SingleTypeKVManager,
        request_id: str,
        block_ids: tuple[int, ...],
        context_len: int,
        num_scheduled_tokens: int,
        sparse_config: object,
    ) -> set[int]:
        valid_indices = {
            block_index for block_index, block_id in enumerate(block_ids) if block_id != 0
        }
        if not valid_indices:
            return set()

        states = manager.get_sparse_states(request_id)
        scored_candidates = {
            block_index
            for block_index in valid_indices
            if states.get(block_index) is not None
            and states[block_index].ema_score is not None
            and not states[block_index].is_permanently_evicted
        }
        if not scored_candidates:
            return valid_indices

        protected = self._get_protected_block_indices(
            num_blocks=len(block_ids),
            context_len=context_len,
            num_scheduled_tokens=num_scheduled_tokens,
            block_size=manager.block_size,
            sparse_config=sparse_config,
        )
        keep_budget = max(
            len(protected & valid_indices),
            ceil(len(valid_indices) * (1.0 - getattr(sparse_config, "compression_ratio", 0.5))),
        )
        if keep_budget >= len(valid_indices):
            return valid_indices

        keep_indices = set(protected & valid_indices)
        ranked_candidates = sorted(
            (block_index for block_index in valid_indices if block_index not in keep_indices),
            key=lambda block_index: states.get(block_index).ema_score
            if states.get(block_index) is not None
            and states[block_index].ema_score is not None
            else float("inf"),
            reverse=True,
        )
        keep_indices.update(ranked_candidates[: max(0, keep_budget - len(keep_indices))])
        return keep_indices

    def _get_protected_block_indices(
        self,
        *,
        num_blocks: int,
        context_len: int,
        num_scheduled_tokens: int,
        block_size: int,
        sparse_config: object,
    ) -> set[int]:
        protected: set[int] = set()
        sink_blocks = min(getattr(sparse_config, "prefix_sink_blocks", 1), num_blocks)
        protected.update(range(sink_blocks))
        recent_blocks = min(getattr(sparse_config, "protected_recent_blocks", 2), num_blocks)
        protected.update(range(max(0, num_blocks - recent_blocks), num_blocks))
        if num_blocks > 0:
            current_start = context_len // block_size
            current_end = (context_len + max(num_scheduled_tokens, 1) - 1) // block_size
            protected.update(range(current_start, min(current_end + 1, num_blocks)))
            if (context_len + num_scheduled_tokens) % block_size != 0:
                protected.add(num_blocks - 1)
        return protected

__all__ = [
    "KVCacheBlocks",
    "KVManager",
    "KVManagerConfig",
    "LayerBlockSelection",
]
