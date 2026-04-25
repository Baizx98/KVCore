from __future__ import annotations

import torch
from torch import nn

from kvcore.model.attn_backend import AttentionBackend, AttentionType, build_attention_backend
from kvcore.model.forward_context import get_forward_context
from kvcore.model.model_utils import extract_layer_index


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
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = self.head_size if head_size_v is None else head_size_v
        self.scale = scale
        self.layer_idx = extract_layer_index(prefix)
        self.sliding_window = sliding_window
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_kv_heads, "
                f"got {self.num_heads=} and {self.num_kv_heads=}"
            )

        self.backend_impl = build_attention_backend(attn_backend)

    def get_attn_backend(self) -> AttentionBackend:
        return self.backend_impl

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        output_shape: torch.Size | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        output_dtype = query.dtype
        if output_shape is None:
            num_tokens = query.shape[0]
            output_shape = torch.Size((num_tokens, self.num_heads * self.head_size_v))
        else:
            output_shape = torch.Size(output_shape)
        hidden_size = output_shape[-1]
        if hidden_size != self.num_heads * self.head_size_v:
            raise ValueError(
                "Invalid attention output shape. "
                f"Expected last dim {self.num_heads * self.head_size_v}, got {hidden_size}"
            )

        output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        query = query.view(-1, self.num_heads, self.head_size)
        output_view = output.view(-1, self.num_heads, self.head_size_v)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size_v)

        attn_metadata = get_forward_context().attn_metadata
        output_view = self.backend_impl.forward(
            query,
            key,
            value,
            num_kv_heads=self.num_kv_heads,
            scaling=self.scale,
            is_causal=self.attn_type == AttentionType.DECODER,
            attn_metadata=attn_metadata,
            layer_idx=self.layer_idx,
            output=output_view,
        )
        output = output_view.reshape(*tuple(output_shape))
        return output

    def extra_repr(self) -> str:
        return (
            f"head_size={self.head_size}, num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, scale={self.scale}, "
            f"backend={self.backend_impl.__class__.__name__}"
        )
