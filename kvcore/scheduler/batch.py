"""Scheduled batch definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from kvcore.scheduler.request_state import RequestState


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
