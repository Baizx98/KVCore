from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

import torch


class AttentionType(StrEnum):
    DECODER = "decoder"


@runtime_checkable
class AttentionBackend(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        num_kv_heads: int,
        scaling: float,
        is_causal: bool,
        attn_metadata: object | None = None,
        layer_idx: int | None = None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


__all__ = ["AttentionBackend", "AttentionType"]
