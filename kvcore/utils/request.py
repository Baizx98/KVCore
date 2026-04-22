from __future__ import annotations

import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from kvcore.utils.sampling_params import SamplingParams


class FinishReason(enum.StrEnum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"
    ERROR = "error"


class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_ERROR = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: RequestStatus) -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: RequestStatus) -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
}


@dataclass(frozen=True, slots=True)
class RequestEvent:
    event_type: str
    timestamp: float

    @classmethod
    def new_event(cls, event_type: str, timestamp: float | None = None) -> RequestEvent:
        return cls(event_type=event_type, timestamp=time.time() if timestamp is None else timestamp)


@dataclass(slots=True)
class StreamingUpdate:
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None

    @classmethod
    def from_request(cls, request: Request) -> StreamingUpdate | None:
        if not request.resumable:
            return None
        return cls(
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
        )


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        *,
        client_index: int = 0,
        arrival_time: float | None = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[[Request], list[Any]] | None = None,
        resumable: bool = False,
    ) -> None:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")

        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.arrival_time = time.time() if arrival_time is None else arrival_time
        self.status = RequestStatus.WAITING
        self.stop_reason: int | str | None = None
        self.events: list[RequestEvent] = []
        self.kv_transfer_params: dict[str, Any] | None = None

        self.max_tokens = sampling_params.max_tokens
        if sampling_params.extra_args is not None:
            params = sampling_params.extra_args.get("kv_transfer_params")
            self.kv_transfer_params = params if isinstance(params, dict) else None

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = list(prompt_token_ids)
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.num_output_placeholders = 0
        self.discard_latest_async_tokens = False
        self.cache_salt = cache_salt
        self.trace_headers = trace_headers
        self.is_prefill_chunk = False
        self.num_nans_in_logits = 0
        self.num_preemptions = 0
        self.prefill_stats: dict[str, Any] | None = {}
        self.block_hashes: list[Any] = []
        self._block_hasher = block_hasher
        self.update_block_hashes()
        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        self.resumable = resumable
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        return tuple(self._output_token_ids)

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        return tuple(self._all_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)
        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    def get_skip_reading_prefix_cache(self) -> bool:
        if self.sampling_params.skip_reading_prefix_cache is not None:
            return self.sampling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def record_event(
        self,
        event_type: str,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(RequestEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[RequestEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def take_prefill_stats(self) -> dict[str, Any] | None:
        if self.prefill_stats is None:
            return None
        prefill_stats = self.prefill_stats
        self.prefill_stats = None
        return prefill_stats

    def mark_finished(
        self,
        status: RequestStatus,
        *,
        stop_reason: int | str | None = None,
    ) -> None:
        if not RequestStatus.is_finished(status):
            raise ValueError(f"status must be a finished status, got {status}")
        self.status = status
        self.stop_reason = stop_reason

    def __lt__(self, other: Request) -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)
