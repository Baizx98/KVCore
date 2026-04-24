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
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
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
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self._canonicalize_positions(
            positions,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        freqs = positions[..., None].to(torch.float32) * self.inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    @staticmethod
    def _canonicalize_positions(
        positions: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions = positions.to(device=device)
        if positions.dim() == 1:
            if positions.numel() != seq_len:
                raise ValueError(
                    f"1D positions must have seq_len={seq_len}, got {positions.numel()}"
                )
            return positions.unsqueeze(0).expand(batch_size, -1)

        if positions.dim() == 2:
            if positions.shape == (batch_size, seq_len):
                return positions
            if positions.shape == (1, seq_len):
                return positions.expand(batch_size, -1)
            raise ValueError(
                "2D positions must have shape "
                f"({batch_size}, {seq_len}) or (1, {seq_len}), got {tuple(positions.shape)}"
            )

        raise ValueError(f"positions must be 1D or 2D, got shape {tuple(positions.shape)}")
