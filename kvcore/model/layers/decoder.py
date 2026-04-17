"""Decoder block layers."""

from __future__ import annotations

import torch
from torch import nn

from kvcore.model.layers.activation import SiluAndMul
from kvcore.model.layers.attention import Attention, BlockedKVCacheCollection
from kvcore.model.layers.linear import MergedColumnLinear, RowLinear
from kvcore.model.layers.norm import RMSNorm
from kvcore.model.metadata import AttentionMetadata


class DecoderMLP(nn.Module):
    """vLLM-style gated MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=bias,
        )
        self.down_proj = RowLinear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class DecoderLayer(nn.Module):
    """One decoder layer following vLLM's high-level structure."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float,
        rms_norm_eps: float,
        attention_bias: bool,
        mlp_bias: bool,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
            bias=attention_bias,
            layer_idx=layer_idx,
        )
        self.mlp = DecoderMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=mlp_bias,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        kv_caches: BlockedKVCacheCollection,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, kv_caches, attn_metadata)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
