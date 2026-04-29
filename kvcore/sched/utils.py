from __future__ import annotations

from dataclasses import dataclass, field

from kvcore.utils.request import FinishReason
from kvcore.utils.sampling_params import SamplingParams


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_num_seqs: int = 8
    max_num_scheduled_tokens: int = 512
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    long_prefill_token_threshold: int = 0
    reserve_full_prompt_blocks: bool = False

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
        if self.max_num_partial_prefills <= 0:
            raise ValueError(
                "max_num_partial_prefills must be positive, "
                f"got {self.max_num_partial_prefills}"
            )
        if self.max_long_partial_prefills <= 0:
            raise ValueError(
                "max_long_partial_prefills must be positive, "
                f"got {self.max_long_partial_prefills}"
            )
        if self.long_prefill_token_threshold < 0:
            raise ValueError(
                "long_prefill_token_threshold must be non-negative, "
                f"got {self.long_prefill_token_threshold}"
            )


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    request_id: str
    is_prefill: bool
    context_len: int
    query_len: int
    sample_index: int | None
    should_sample: bool
    num_computed_tokens: int = 0
    block_ids: tuple[tuple[int, ...], ...] = ()

    @property
    def num_scheduled_tokens(self) -> int:
        return self.query_len


@dataclass(frozen=True, slots=True)
class NewRequestData:
    req_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_params: SamplingParams
    block_ids: tuple[tuple[int, ...], ...]
    num_computed_tokens: int
    num_scheduled_tokens: int
    is_prefill: bool
    should_sample: bool
    sample_index: int | None


@dataclass(frozen=True, slots=True)
class CachedRequestData:
    req_ids: tuple[str, ...]
    new_token_ids: tuple[tuple[int, ...], ...]
    block_ids: tuple[tuple[tuple[int, ...], ...], ...]
    num_computed_tokens: tuple[int, ...]
    num_scheduled_tokens: tuple[int, ...]
    is_prefill: tuple[bool, ...]
    should_sample: tuple[bool, ...]
    sample_indices: tuple[int | None, ...]

    @classmethod
    def empty(cls) -> CachedRequestData:
        return cls(
            req_ids=(),
            new_token_ids=(),
            block_ids=(),
            num_computed_tokens=(),
            num_scheduled_tokens=(),
            is_prefill=(),
            should_sample=(),
            sample_indices=(),
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    scheduled_requests: tuple[ScheduledRequest, ...]
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    num_prefill_reqs: int
    num_decode_reqs: int
    new_block_ids_to_zero: tuple[int, ...] = ()
    scheduled_new_reqs: tuple[NewRequestData, ...] = ()
    scheduled_cached_reqs: CachedRequestData = field(default_factory=CachedRequestData.empty)

    @property
    def is_empty(self) -> bool:
        return self.total_num_scheduled_tokens == 0

    @classmethod
    def empty(cls) -> SchedulerOutput:
        return cls(
            scheduled_requests=(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            num_prefill_reqs=0,
            num_decode_reqs=0,
            new_block_ids_to_zero=(),
            scheduled_new_reqs=(),
            scheduled_cached_reqs=CachedRequestData.empty(),
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
    "CachedRequestData",
    "NewRequestData",
    "ScheduledRequest",
    "SchedulerConfig",
    "SchedulerOutput",
    "SchedulerUpdateResult",
]
