from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SiluAndMul(nn.Module):
    def forward(
        self,
        gate: torch.Tensor,
        up: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if up is None:
            gate, up = gate.chunk(2, dim=-1)
        return F.silu(gate) * up

