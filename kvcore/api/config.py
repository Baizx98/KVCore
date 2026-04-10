"""Configuration objects for KVCore."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

DEFAULT_MODEL_PATH = "~/Tan/model/Llama-3.1-8B-Instruct"


def _normalize_eos_token_ids(value: int | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    normalized = tuple(int(token_id) for token_id in value)
    if not normalized:
        raise ValueError("eos_token_id sequence must not be empty")
    return normalized


@dataclass(slots=True)
class EngineConfig:
    """Configuration for model loading and runtime defaults."""

    model_name_or_path: str = DEFAULT_MODEL_PATH
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 32
    trust_remote_code: bool = False
    block_size: int = 16
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


@dataclass(slots=True)
class GenerationConfig:
    """Generation settings for the minimal greedy path."""

    max_new_tokens: int | None = None
    eos_token_id: int | Sequence[int] | None = None
    pad_token_id: int | None = None
    skip_special_tokens: bool = True
    stop_on_eos: bool = True
    normalized_eos_token_ids: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive when provided")
        self.normalized_eos_token_ids = _normalize_eos_token_ids(self.eos_token_id)
