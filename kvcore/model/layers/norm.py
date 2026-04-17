"""Normalization layers."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """vLLM-style RMSNorm supporting an optional residual path."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = hidden_states
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        if residual is not None:
            return hidden_states, residual
        return hidden_states
