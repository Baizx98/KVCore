from __future__ import annotations

from dataclasses import dataclass

import torch

from kvcore.kv.block_table import MultiGroupBlockTable


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
]
