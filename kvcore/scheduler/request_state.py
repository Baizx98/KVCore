"""Per-request scheduler state."""

from __future__ import annotations

from dataclasses import dataclass, field

from kvcore.api.types import Request


@dataclass(slots=True)
class RequestState:
    """Track one request as it moves through waiting and running."""

    request: Request
    status: str = "waiting"
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    encoded_inputs: dict[str, object] | None = None
    past_key_values: object | None = None
    sequence_state: object | None = None
    finished: bool = False
    finish_reason: str | None = None

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def prompt(self) -> str:
        return self.request.prompt

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)
