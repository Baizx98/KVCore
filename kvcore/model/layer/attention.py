from __future__ import annotations

import torch
from torch import nn

from kvcore.model.attn_backend import AttentionBackend, AttentionType, build_attention_backend


class Attention(nn.Module):
    """A thin PyTorch attention wrapper around the configured backend.

    This mirrors the role of vLLM's `Attention` layer at a smaller scope:
    model-specific modules prepare q/k/v tensors, while this layer owns the
    backend selection, optional bound KV cache, and output shape restoration.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        *,
        num_kv_heads: int | None = None,
        prefix: str = "",
        attn_type: AttentionType = AttentionType.DECODER,
        attn_backend: str | AttentionBackend | None = None,
        head_size_v: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = self.head_size if head_size_v is None else head_size_v
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_kv_heads, "
                f"got {self.num_heads=} and {self.num_kv_heads=}"
            )

        self.backend_impl = build_attention_backend(attn_backend)
        self.kv_cache: object | None = None

    def bind_kv_cache(self, kv_cache: object | None) -> None:
        self.kv_cache = kv_cache

    def get_attn_backend(self) -> AttentionBackend:
        return self.backend_impl

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        output_shape: torch.Size | tuple[int, ...] | None = None,
        attn_metadata: object | None = None,
        kv_cache: object | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        query, key, value, original_rank = self._canonicalize_qkv(query, key, value)
        attn_output = self.backend_impl.forward(
            query,
            key,
            value,
            num_kv_heads=self.num_kv_heads,
            scaling=self.scale,
            is_causal=self.attn_type == AttentionType.DECODER,
            attn_metadata=attn_metadata,
            kv_cache=self.kv_cache if kv_cache is None else kv_cache,
            layer_idx=layer_idx,
        )

        if output_shape is None:
            output_shape = self._infer_output_shape(
                query=query,
                original_rank=original_rank,
            )

        hidden_size = self.num_heads * self.head_size_v
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*tuple(output_shape))
        if attn_output.shape[-1] != hidden_size:
            raise ValueError(
                "Invalid attention output shape. "
                f"Expected last dim {hidden_size}, got {attn_output.shape[-1]}"
            )
        return attn_output

    def extra_repr(self) -> str:
        return (
            f"head_size={self.head_size}, num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, scale={self.scale}, "
            f"backend={self.backend_impl.__class__.__name__}"
        )

    def _canonicalize_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if query.dim() == 4:
            return query, key, value, 4
        if query.dim() == 3:
            query = query.transpose(0, 1).unsqueeze(0)
            key = key.transpose(0, 1).unsqueeze(0)
            value = value.transpose(0, 1).unsqueeze(0)
            return query, key, value, 3
        raise ValueError(
            "query/key/value must be rank-3 or rank-4 tensors, "
            f"got query shape {tuple(query.shape)}"
        )

    def _infer_output_shape(
        self,
        *,
        query: torch.Tensor,
        original_rank: int,
    ) -> torch.Size:
        hidden_size = self.num_heads * self.head_size_v
        if original_rank == 3:
            return torch.Size((query.size(2), hidden_size))
        return torch.Size((query.size(0), query.size(2), hidden_size))
