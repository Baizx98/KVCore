from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from kvcore.kv.block_table import MultiGroupBlockTable
from kvcore.kv.single_type_kv_manager import LayerBlockSelection


@dataclass(frozen=True, slots=True)
class SparseComputePlan:
    selections: tuple[LayerBlockSelection, ...]

    @classmethod
    def from_selections(
        cls,
        selections: Sequence[LayerBlockSelection],
    ) -> SparseComputePlan:
        return cls(tuple(selections))

    def get_skip_indices(self, request_id: str, layer_idx: int) -> set[int]:
        skip_indices: set[int] = set()
        for selection in self.selections:
            if selection.request_id == request_id and selection.layer_idx == layer_idx:
                skip_indices.update(selection.block_indices)
        return skip_indices


@dataclass(frozen=True, slots=True)
class KVForwardMetadata:
    block_tables: MultiGroupBlockTable
    slot_mapping: dict[int, torch.Tensor]
    kv_cache_tensor: torch.Tensor | None = None
    sparse_plan: SparseComputePlan | None = None
    logical_block_indices: dict[int, tuple[tuple[int, ...], ...]] | None = None
    skipped_block_indices: dict[int, tuple[tuple[int, ...], ...]] | None = None

__all__ = [
    "KVForwardMetadata",
    "SparseComputePlan",
]
