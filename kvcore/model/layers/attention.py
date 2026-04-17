"""Attention layer and blocked KV cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from kvcore.model.layers.linear import QKVParallelLinear, RowLinear
from kvcore.model.layers.rotary import RotaryEmbedding
from kvcore.model.metadata import AttentionMetadata


def install_attention_hook_state(*, attention_module: Any, attention_params: Any) -> None:
    """Attach KVCore attention metadata to one attention module."""

    attention_module.kvcore_attention_params = attention_params
    attention_module.kvcore_block_table = attention_params.block_table


def clear_attention_hook_state(*, attention_module: Any) -> None:
    """Clear per-step KVCore metadata from one attention module."""

    for attr_name in ("kvcore_attention_params", "kvcore_block_table"):
        if hasattr(attention_module, attr_name):
            delattr(attention_module, attr_name)


@dataclass(slots=True)
class BlockedKVCache:
    """One layer of blocked KV cache."""

    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_size: int

    @classmethod
    def allocate(
        cls,
        *,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> BlockedKVCache:
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        return cls(
            key_cache=torch.zeros(shape, dtype=dtype, device=device),
            value_cache=torch.zeros(shape, dtype=dtype, device=device),
            block_size=block_size,
        )

    def write(
        self,
        *,
        slot_mapping: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        for token_idx, slot in enumerate(slot_mapping.tolist()):
            block_idx = slot // self.block_size
            block_offset = slot % self.block_size
            self.key_cache[block_idx, block_offset] = key_states[token_idx]
            self.value_cache[block_idx, block_offset] = value_states[token_idx]

    def gather(
        self,
        *,
        block_table: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gathered_keys: list[torch.Tensor] = []
        gathered_values: list[torch.Tensor] = []
        remaining = seq_len
        for block_id in block_table.tolist():
            if remaining <= 0 or block_id < 0:
                break
            take = min(self.block_size, remaining)
            gathered_keys.append(self.key_cache[block_id, :take])
            gathered_values.append(self.value_cache[block_id, :take])
            remaining -= take
        if not gathered_keys:
            shape = (0, self.key_cache.shape[2], self.key_cache.shape[3])
            empty = self.key_cache.new_zeros(shape)
            return empty, empty
        return torch.cat(gathered_keys, dim=0), torch.cat(gathered_values, dim=0)


@dataclass(slots=True)
class BlockedKVCacheCollection:
    """KV cache bundle across all decoder layers."""

    layers: list[BlockedKVCache]
    current_seq_len: int = 0

    def get_seq_length(self) -> int:
        return self.current_seq_len


class Attention(nn.Module):
    """Manual blocked attention over flattened tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        rope_theta: float,
        bias: bool,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_heads = self.total_num_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
        )
        self.o_proj = RowLinear(self.total_num_heads * self.head_dim, hidden_size, bias=bias)
        self.rotary_emb = RotaryEmbedding(head_dim=self.head_dim, rope_theta=rope_theta)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_caches: BlockedKVCacheCollection,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        kv_cache = kv_caches.layers[self.layer_idx]
        kv_cache.write(slot_mapping=attn_metadata.slot_mapping, key_states=k, value_states=v)
        kv_caches.current_seq_len = attn_metadata.max_seq_len

        outputs: list[torch.Tensor] = []
        for seq_idx in range(attn_metadata.num_sequences):
            start = int(attn_metadata.query_start_locs[seq_idx].item())
            end = int(attn_metadata.query_start_locs[seq_idx + 1].item())
            query = q[start:end]
            seq_len = int(attn_metadata.seq_lens[seq_idx].item())
            context_len = int(attn_metadata.context_lens[seq_idx].item())
            cached_k, cached_v = kv_cache.gather(
                block_table=attn_metadata.block_tables[seq_idx],
                seq_len=seq_len,
            )
            cached_k = repeat_kv(cached_k, self.num_heads // self.num_kv_heads)
            cached_v = repeat_kv(cached_v, self.num_heads // self.num_kv_heads)
            outputs.append(
                _run_sequence_attention(
                    query=query,
                    key=cached_k,
                    value=cached_v,
                    context_len=context_len,
                    scale=self.scaling,
                )
            )

        attn_output = torch.cat(outputs, dim=0)
        attn_output = attn_output.reshape(attn_output.shape[0], self.hidden_size)
        output, _ = self.o_proj(attn_output)
        return output


def repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    if num_repeats == 1:
        return hidden_states
    seq_len, num_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :].expand(seq_len, num_heads, num_repeats, head_dim)
    return hidden_states.reshape(seq_len, num_heads * num_repeats, head_dim)


def _run_sequence_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    context_len: int,
    scale: float,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    query_len = query.shape[0]
    for token_idx in range(query_len):
        visible = context_len + token_idx + 1
        q = query[token_idx]
        k = key[:visible]
        v = value[:visible]
        attn_scores = torch.einsum("hd,thd->ht", q, k) * scale
        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(q.dtype)
        outputs.append(torch.einsum("ht,thd->hd", attn_probs, v))
    return torch.stack(outputs, dim=0)
