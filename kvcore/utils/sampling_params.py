from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None
    skip_reading_prefix_cache: bool | None = None
    extra_args: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive when set, got {self.top_k}")
