"""Linear layers with vLLM-like packing semantics."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MergedColumnLinear(nn.Module):
    """A single linear layer that packs multiple column-wise projections."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        self.output_sizes = output_sizes
        self.linear = nn.Linear(input_size, sum(output_sizes), bias=bias)

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.linear.bias

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.linear(hidden_states), None


class QKVParallelLinear(nn.Module):
    """Pack q/k/v projections in one linear layer."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        self.q_size = total_num_heads * head_size
        self.kv_size = total_num_kv_heads * head_size
        self.linear = nn.Linear(hidden_size, self.q_size + (2 * self.kv_size), bias=bias)

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.linear.bias

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.linear(hidden_states), None


class RowLinear(nn.Module):
    """Simple row-parallel replacement for local single-device execution."""

    def __init__(self, input_size: int, output_size: int, *, bias: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.linear.bias

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.linear(hidden_states), None
