from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

from kvcore.sched.utils import FinishedRequestState, SchedulerOutput, SchedulerUpdateResult
from kvcore.utils.request import Request, RequestStatus


class SchedulerInterface(ABC):
    @abstractmethod
    def schedule(self) -> SchedulerOutput:
        raise NotImplementedError

    @abstractmethod
    def update_from_outputs(
        self,
        scheduler_output: SchedulerOutput,
        sampled_token_ids: Mapping[str, int],
        *,
        stop_token_ids: set[int] | None = None,
    ) -> SchedulerUpdateResult:
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: Request) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> tuple[FinishedRequestState, ...]:
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        raise NotImplementedError

    def has_unfinished_requests(self) -> bool:
        return self.get_num_unfinished_requests() > 0

    @abstractmethod
    def has_finished_requests(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError


__all__ = [
    "SchedulerInterface",
]
