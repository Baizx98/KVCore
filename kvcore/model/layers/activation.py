"""Activation layers aligned with vLLM-style decoder blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    """Apply SiLU to the first half and multiply by the second half."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = hidden_states.chunk(2, dim=-1)
        return F.silu(gate) * up
