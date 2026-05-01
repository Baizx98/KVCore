from __future__ import annotations

from kvcore.sched.request_queue import RequestQueue
from kvcore.utils.request import Request
from kvcore.utils.sampling_params import SamplingParams


def make_request(request_id: str) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
    )


def request_ids(queue: RequestQueue) -> tuple[str, ...]:
    return tuple(request.request_id for request in queue)


def test_request_queue_pop_is_fcfs() -> None:
    queue = RequestQueue()
    queue.add_request(make_request("a"))
    queue.add_request(make_request("b"))

    assert queue.pop_request().request_id == "a"
    assert queue.pop_request().request_id == "b"
    assert not queue


def test_request_queue_prepend_preserves_order() -> None:
    queue = RequestQueue()
    queue.add_request(make_request("c"))
    queue.prepend_request(make_request("b"))
    queue.prepend_requests((make_request("a0"), make_request("a1")))

    assert request_ids(queue) == ("a0", "a1", "b", "c")


def test_request_queue_remove_requests() -> None:
    queue = RequestQueue()
    requests = [make_request("a"), make_request("b"), make_request("c")]
    for request in requests:
        queue.add_request(request)

    assert queue.remove_request(requests[1])
    queue.remove_requests((requests[0],))

    assert request_ids(queue) == ("c",)


def test_request_queue_peek_does_not_pop() -> None:
    queue = RequestQueue()
    queue.add_request(make_request("a"))

    assert queue.peek_request().request_id == "a"
    assert queue.peek_request().request_id == "a"
    assert len(queue) == 1
