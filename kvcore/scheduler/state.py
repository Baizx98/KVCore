"""Scheduler-owned request state and batch descriptions."""

from __future__ import annotations

from dataclasses import dataclass, field

from kvcore.api.types import Request
from kvcore.model_runner.context import SequenceState


@dataclass(slots=True)
class RequestState:
    """Track one request as it moves through waiting and running."""

    request: Request
    status: str = "waiting"
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    encoded_inputs: dict[str, object] | None = None
    past_key_values: object | None = None
    sequence_state: SequenceState | None = None
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


@dataclass(slots=True)
class ScheduledBatch:
    """A scheduled prefill or decode batch."""

    mode: str
    request_states: list[RequestState]
    request_ids: list[str]
    num_requests: int
    num_tokens: int
    flat_input_ids: list[int]
    flat_position_ids: list[int]
    request_offsets: list[int]
    request_token_counts: list[int]
    encoded_inputs: dict[str, object] | None = None
    metadata: dict[str, int | str | bool] = field(default_factory=dict)
