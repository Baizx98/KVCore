from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif query.dim() == 4:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        raise ValueError(
            "query/key must be rank-3 [tokens, heads, head_dim] or "
            f"rank-4 [batch, heads, seq, head_dim], got {tuple(query.shape)}"
        )
    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)
    return query, key


def _resolve_rope_base(config: PretrainedConfig) -> float:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, Mapping):
        if "rope_theta" in rope_parameters:
            return float(rope_parameters["rope_theta"])
        if "theta" in rope_parameters:
            return float(rope_parameters["theta"])

    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        return float(rope_theta)
    return 10000.0


class RotaryEmbedding(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
        self.base = _resolve_rope_base(config)

        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.dim() != 2 or key.dim() != 2:
            raise ValueError(
                "RotaryEmbedding expects flattened q/k tensors with shape "
                f"[num_tokens, hidden], got {tuple(query.shape)} and {tuple(key.shape)}"
            )
        positions = positions.to(device=query.device)
        if positions.dim() != 1 or positions.numel() != query.size(0):
            raise ValueError(
                "positions must be rank-1 with one entry per token, "
                f"got positions shape {tuple(positions.shape)} for {query.size(0)} tokens"
            )
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(query.size(0), -1, self.head_dim)
        key = key.view(key.size(0), -1, self.head_dim)
        freqs = positions[:, None].to(torch.float32) * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=query.dtype)
        sin = emb.sin().to(dtype=query.dtype)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        return query.reshape(query_shape), key.reshape(key_shape)
