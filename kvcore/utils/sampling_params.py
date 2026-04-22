from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_tokens: int
    skip_reading_prefix_cache: bool | None = None
    extra_args: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

