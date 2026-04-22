from __future__ import annotations

import pytest

from kvcore.utils.request import (
    FinishReason,
    Request,
    RequestStatus,
    StreamingUpdate,
)
from kvcore.utils.sampling_params import SamplingParams


def test_request_initializes_generation_state() -> None:
    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        arrival_time=10.0,
    )

    assert request.request_id == "req-1"
    assert request.status == RequestStatus.WAITING
    assert request.max_tokens == 4
    assert request.num_prompt_tokens == 3
    assert request.num_tokens == 3
    assert request.output_token_ids == ()
    assert request.all_token_ids == (1, 2, 3)
    assert not request.is_finished()


def test_request_appends_output_tokens_and_updates_counts() -> None:
    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=4),
    )

    request.append_output_token_ids(3)
    request.append_output_token_ids([4, 5])

    assert request.output_token_ids == (3, 4, 5)
    assert request.all_token_ids == (1, 2, 3, 4, 5)
    assert request.num_output_tokens == 3
    assert request.num_tokens == 5


def test_request_rejects_empty_prompt_token_ids() -> None:
    with pytest.raises(ValueError, match="prompt_token_ids must be non-empty"):
        Request(
            request_id="bad",
            prompt_token_ids=[],
            sampling_params=SamplingParams(max_tokens=1),
        )


def test_request_reads_sampling_skip_prefix_cache_flag() -> None:
    request = Request(
        request_id="req-1",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=1, skip_reading_prefix_cache=True),
    )

    assert request.skip_reading_prefix_cache is True


def test_request_events_are_consumed_once() -> None:
    request = Request(
        request_id="req-1",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=1),
    )

    assert request.take_events() is None
    request.record_event("scheduled", timestamp=123.0)
    events = request.take_events()

    assert events is not None
    assert len(events) == 1
    assert events[0].event_type == "scheduled"
    assert events[0].timestamp == 123.0
    assert request.take_events() is None


def test_request_finished_status_and_reason() -> None:
    request = Request(
        request_id="req-1",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=1),
    )

    request.mark_finished(RequestStatus.FINISHED_LENGTH_CAPPED)

    assert request.is_finished()
    assert request.get_finished_reason() == FinishReason.LENGTH


def test_request_priority_ordering() -> None:
    high_priority = Request(
        request_id="b",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=1),
        priority=-1,
        arrival_time=2.0,
    )
    low_priority = Request(
        request_id="a",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=1),
        priority=0,
        arrival_time=1.0,
    )

    assert high_priority < low_priority


def test_streaming_update_from_resumable_request() -> None:
    sampling_params = SamplingParams(max_tokens=3)
    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2],
        sampling_params=sampling_params,
        arrival_time=12.0,
        resumable=True,
    )

    update = StreamingUpdate.from_request(request)

    assert update is not None
    assert update.prompt_token_ids == [1, 2]
    assert update.max_tokens == 3
    assert update.arrival_time == 12.0
    assert update.sampling_params is sampling_params


def test_block_hasher_is_called_on_updates() -> None:
    calls = 0

    def block_hasher(request: Request) -> list[str]:
        nonlocal calls
        calls += 1
        return [f"hash-{request.num_tokens}"]

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=3),
        block_hasher=block_hasher,
    )
    request.append_output_token_ids([2, 3])

    assert calls == 2
    assert request.block_hashes == ["hash-1", "hash-3"]
