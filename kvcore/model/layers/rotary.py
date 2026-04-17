"""Rotary embedding helpers."""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary position embedding over flattened token positions."""

    def __init__(self, *, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin(positions)
        return apply_rotary_pos_emb(query, key, cos, sin)

    def cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[:, None, :].to(query.dtype)
    sin = sin[:, None, :].to(query.dtype)
    return (
        (query * cos) + (rotate_half(query) * sin),
        (key * cos) + (rotate_half(key) * sin),
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
