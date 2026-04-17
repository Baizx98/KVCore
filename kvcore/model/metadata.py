"""Model-side metadata for flattened batches and blocked KV cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ModelKVCacheSpec:
    """Describe how one model family organizes its KV cache."""

    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    block_size: int
    cache_dtype: str
    attention_dtype: str
    use_sliding_window: bool = False
    use_cross_attention: bool = False


@dataclass(slots=True)
class AttentionMetadata:
    """Flattened attention metadata inspired by vLLM's step inputs."""

    query_start_locs: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    max_query_len: int
    max_seq_len: int
    num_prefill_sequences: int
    num_decode_tokens: int

    def __post_init__(self) -> None:
        if self.query_start_locs.numel() != self.seq_lens.numel() + 1:
            raise ValueError("query_start_locs must have len(seq_lens) + 1 entries")
        if self.seq_lens.numel() != self.context_lens.numel():
            raise ValueError("seq_lens and context_lens must have identical length")
        if self.slot_mapping.numel() != int(self.query_start_locs[-1].item()):
            raise ValueError("slot_mapping size must equal flattened query token count")

    @property
    def num_sequences(self) -> int:
        return int(self.seq_lens.numel())

    @property
    def num_query_tokens(self) -> int:
        return int(self.query_start_locs[-1].item())

    @property
    def query_lens(self) -> torch.Tensor:
        return self.query_start_locs[1:] - self.query_start_locs[:-1]

    @property
    def is_chunked_prefill(self) -> bool:
        return bool(self.num_prefill_sequences > 0 and torch.any(self.context_lens > 0))

    @property
    def is_decode_only(self) -> bool:
        return self.num_prefill_sequences == 0 and self.num_decode_tokens > 0

    @property
    def is_prefill_only(self) -> bool:
        return self.num_prefill_sequences == self.num_sequences and self.num_decode_tokens == 0
