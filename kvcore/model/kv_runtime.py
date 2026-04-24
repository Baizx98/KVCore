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
class PagedAttentionMetadata:
    kv_cache_tensor: torch.Tensor
    block_tables: MultiGroupBlockTable
    slot_mapping: dict[int, torch.Tensor]
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    query_lens: torch.Tensor
    flat_positions: torch.Tensor
    token_request_indices: torch.Tensor
    num_reqs: int
    num_scheduled_tokens: int
    num_prefill_reqs: int
    num_decode_reqs: int
    max_query_len: int
    max_seq_len: int


__all__ = [
    "PagedAttentionMetadata",
    "SparseComputePlan",
]
