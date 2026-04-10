"""Public data types used by KVCore."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Request:
    """A single text-generation request."""

    prompt: str
    request_id: str = "req-0"

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if not self.request_id:
            raise ValueError("request_id must not be empty")


@dataclass(slots=True)
class GenerationResult:
    """Result returned by the minimal generation API."""

    text: str
    token_ids: list[int]
    generated_token_ids: list[int]
    finish_reason: str
    num_prompt_tokens: int
    num_generated_tokens: int
    request_id: str
    kv_block_count: int
    kv_total_tokens: int
    metadata: dict[str, str | int | bool] = field(default_factory=dict)
