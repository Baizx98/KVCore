from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

from kvcore.utils.request import Request


class RequestQueue:
    """FCFS request queue with vLLM-style prepend helpers."""

    def __init__(self) -> None:
        self._queue: deque[Request] = deque()

    def add_request(self, request: Request) -> None:
        self._queue.append(request)

    def pop_request(self) -> Request:
        return self._queue.popleft()

    def peek_request(self) -> Request | None:
        if not self._queue:
            return None
        return self._queue[0]

    def prepend_request(self, request: Request) -> None:
        self._queue.appendleft(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        for request in reversed(tuple(requests)):
            self._queue.appendleft(request)

    def remove_request(self, request: Request) -> bool:
        try:
            self._queue.remove(request)
        except ValueError:
            return False
        return True

    def remove_requests(self, requests: Iterable[Request]) -> None:
        request_ids = {request.request_id for request in requests}
        self._queue = deque(
            request for request in self._queue if request.request_id not in request_ids
        )

    def __iter__(self) -> Iterator[Request]:
        return iter(self._queue)

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)


__all__ = [
    "RequestQueue",
]
