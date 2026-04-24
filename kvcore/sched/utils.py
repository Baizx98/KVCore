from __future__ import annotations

from dataclasses import dataclass

from kvcore.utils.request import FinishReason
from kvcore.utils.sampling_params import SamplingParams


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_num_seqs: int = 8
    max_num_scheduled_tokens: int = 512

    def __post_init__(self) -> None:
        if self.max_num_seqs <= 0:
            raise ValueError(
                f"max_num_seqs must be positive, got {self.max_num_seqs}"
            )
        if self.max_num_scheduled_tokens <= 0:
            raise ValueError(
                "max_num_scheduled_tokens must be positive, "
                f"got {self.max_num_scheduled_tokens}"
            )


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    request_id: str
    is_prefill: bool
    flat_start: int
    flat_end: int
    context_len: int
    query_len: int
    sample_index: int | None
    should_sample: bool


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    scheduled_requests: tuple[ScheduledRequest, ...]
    flat_input_token_ids: tuple[int, ...]
    flat_positions: tuple[int, ...]
    sampling_params: tuple[SamplingParams, ...]
    num_scheduled_tokens: int
    num_prefill_reqs: int
    num_decode_reqs: int

    @property
    def is_empty(self) -> bool:
        return self.num_scheduled_tokens == 0

    @classmethod
    def empty(cls) -> SchedulerOutput:
        return cls(
            scheduled_requests=(),
            flat_input_token_ids=(),
            flat_positions=(),
            sampling_params=(),
            num_scheduled_tokens=0,
            num_prefill_reqs=0,
            num_decode_reqs=0,
        )


@dataclass(frozen=True, slots=True)
class RequestStepOutput:
    request_id: str
    sampled_token_id: int | None
    finished: bool
    finish_reason: FinishReason | None


@dataclass(frozen=True, slots=True)
class FinishedRequestState:
    request_id: str
    output_token_ids: tuple[int, ...]
    finish_reason: FinishReason | None


@dataclass(frozen=True, slots=True)
class SchedulerUpdateResult:
    step_outputs: tuple[RequestStepOutput, ...]
    finished_requests: tuple[FinishedRequestState, ...]
__all__ = [
    "RequestStepOutput",
    "FinishedRequestState",
    "ScheduledRequest",
    "SchedulerConfig",
    "SchedulerOutput",
    "SchedulerUpdateResult",
]
