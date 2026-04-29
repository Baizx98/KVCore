from __future__ import annotations

import random
from dataclasses import dataclass

from kvcore.kv.kv_manager import KVManager
from kvcore.kv.single_type_kv_manager import LayerBlockSelection
from kvcore.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class KVCompressionConfig:
    drop_ratio: float = 0.0
    max_blocks: int | None = None
    seed: int | None = None
    layer_ids: tuple[int, ...] | None = None
    skip_tail_blocks: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.drop_ratio <= 1:
            raise ValueError(f"drop_ratio must be in [0, 1], got {self.drop_ratio}")
        if self.max_blocks is not None and self.max_blocks < 0:
            raise ValueError(f"max_blocks must be non-negative, got {self.max_blocks}")
        if self.skip_tail_blocks < 0:
            raise ValueError(
                f"skip_tail_blocks must be non-negative, got {self.skip_tail_blocks}"
            )


@dataclass(frozen=True, slots=True)
class KVCompressionResult:
    selections: tuple[LayerBlockSelection, ...]
    evicted_block_ids: tuple[int, ...]

    @property
    def num_evicted_blocks(self) -> int:
        return len(self.evicted_block_ids)


class RandomKVBlockCompressor:
    """Random permanent KV block eviction for interface validation."""

    def __init__(self, config: KVCompressionConfig) -> None:
        self.config = config

    def compress(
        self,
        kv_manager: KVManager,
        request_ids: tuple[str, ...],
    ) -> KVCompressionResult:
        logger.info(
            "Random KV compression begin requests=%d drop_ratio=%.3f max_blocks=%s "
            "seed=%s",
            len(request_ids),
            self.config.drop_ratio,
            self.config.max_blocks,
            self.config.seed,
        )
        selections = self.select_blocks(kv_manager, request_ids)
        eviction_result = kv_manager.evict_request_blocks(selections)
        result = KVCompressionResult(
            selections=tuple(selections),
            evicted_block_ids=eviction_result.evicted_block_ids,
        )
        logger.info(
            "Random KV compression done selections=%d evicted_blocks=%d",
            len(result.selections),
            result.num_evicted_blocks,
        )
        return result

    def select_blocks(
        self,
        kv_manager: KVManager,
        request_ids: tuple[str, ...],
    ) -> list[LayerBlockSelection]:
        rng = random.Random(self.config.seed)
        layer_ids = self.config.layer_ids or tuple(range(kv_manager.num_layers))
        selections: list[LayerBlockSelection] = []
        for request_id in request_ids:
            cache_blocks = kv_manager.get_blocks(request_id)
            for layer_idx in layer_ids:
                layer_blocks = cache_blocks.blocks[layer_idx]
                candidate_indices = [
                    block_index
                    for block_index, block in enumerate(layer_blocks)
                    if not block.is_null
                    and block_index < max(0, len(layer_blocks) - self.config.skip_tail_blocks)
                ]
                num_to_drop = int(len(candidate_indices) * self.config.drop_ratio)
                if self.config.max_blocks is not None:
                    num_to_drop = min(num_to_drop, self.config.max_blocks)
                if num_to_drop <= 0:
                    continue
                selected = set(rng.sample(candidate_indices, num_to_drop))
                logger.debug(
                    "Random KV compression selected request_id=%s layer=%d "
                    "candidate_blocks=%d selected=%s",
                    request_id,
                    layer_idx,
                    len(candidate_indices),
                    tuple(sorted(selected)),
                )
                selections.append(
                    LayerBlockSelection(
                        request_id=request_id,
                        layer_idx=layer_idx,
                        block_indices=selected,
                    )
                )
        return selections


__all__ = [
    "KVCompressionConfig",
    "KVCompressionResult",
    "RandomKVBlockCompressor",
]
