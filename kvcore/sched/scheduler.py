from __future__ import annotations

from collections.abc import Mapping

from kvcore.kv.kv_manager import KVManager, KVManagerConfig
from kvcore.kv.kv_metrics import KVCacheMetricsCollector
from kvcore.kv.kv_utils import get_request_block_hasher
from kvcore.sched.utils import (
    FinishedRequestState,
    RequestStepOutput,
    ScheduledRequest,
    SchedulerConfig,
    SchedulerOutput,
    SchedulerUpdateResult,
)
from kvcore.utils.request import FinishReason, Request, RequestStatus


class Scheduler:
    """Owns request scheduling state and logical KV block allocation."""

    def __init__(
        self,
        kv_manager_config: KVManagerConfig,
        scheduler_config: SchedulerConfig | None = None,
        *,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.metrics_collector = metrics_collector
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.kv_manager: KVManager = KVManager(
            kv_manager_config,
            metrics_collector=self.metrics_collector,
        )
        self.requests: dict[str, Request] = {}
        self.waiting: list[str] = []
        self.running: list[str] = []
        self.finished_req_ids: set[str] = set()
        self.request_block_hasher = get_request_block_hasher(
            kv_manager_config.layer_specs[0].block_size
        )

    def add_request(self, request: Request) -> None:
        if request.request_id in self.requests:
            raise ValueError(f"Duplicate request_id: {request.request_id}")
        if request._block_hasher is None:  # type: ignore[attr-defined]
            request._block_hasher = self.request_block_hasher  # type: ignore[attr-defined]
            request.update_block_hashes()
        self.requests[request.request_id] = request
        self.waiting.append(request.request_id)

    def abort_request(self, request_id: str) -> Request | None:
        request = self.requests.pop(request_id, None)
        if request is None:
            return None
        self._remove_request_id(self.waiting, request_id)
        self._remove_request_id(self.running, request_id)
        self.kv_manager.free(request)
        request.mark_finished(RequestStatus.FINISHED_ABORTED)
        self.finished_req_ids.add(request_id)
        return request

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        budget = self.scheduler_config.max_num_scheduled_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        scheduled_requests: list[ScheduledRequest] = []
        flat_input_token_ids: list[int] = []
        flat_positions: list[int] = []
        sampling_params = []

        # 1. running decode requests first.
        for request_id in list(self.running):
            if len(scheduled_requests) >= max_num_seqs or budget <= 0:
                break
            request = self.requests[request_id]
            if not self._is_decode_ready(request):
                continue
            scheduled = self._schedule_request(
                request,
                chunk_len=1,
                flat_input_token_ids=flat_input_token_ids,
                flat_positions=flat_positions,
                sampling_params=sampling_params,
            )
            if scheduled is None:
                break
            scheduled_requests.append(scheduled)
            budget -= scheduled.query_len

        # 2. running prefill continuations.
        for request_id in list(self.running):
            if len(scheduled_requests) >= max_num_seqs or budget <= 0:
                break
            request = self.requests[request_id]
            if self._is_decode_ready(request):
                continue
            remaining = request.num_tokens - request.num_computed_tokens
            if remaining <= 0:
                continue
            chunk_len = min(remaining, budget)
            scheduled = self._schedule_request(
                request,
                chunk_len=chunk_len,
                flat_input_token_ids=flat_input_token_ids,
                flat_positions=flat_positions,
                sampling_params=sampling_params,
            )
            if scheduled is None:
                break
            scheduled_requests.append(scheduled)
            budget -= scheduled.query_len

        # 3. waiting prefill requests.
        for request_id in list(self.waiting):
            if len(scheduled_requests) >= max_num_seqs or budget <= 0:
                break
            request = self.requests[request_id]
            remaining = request.num_tokens - request.num_computed_tokens
            if remaining <= 0:
                continue
            chunk_len = min(remaining, budget)
            scheduled = self._schedule_request(
                request,
                chunk_len=chunk_len,
                flat_input_token_ids=flat_input_token_ids,
                flat_positions=flat_positions,
                sampling_params=sampling_params,
                from_waiting=True,
            )
            if scheduled is None:
                break
            scheduled_requests.append(scheduled)
            budget -= scheduled.query_len

        if not scheduled_requests:
            return SchedulerOutput.empty()

        return SchedulerOutput(
            scheduled_requests=tuple(scheduled_requests),
            flat_input_token_ids=tuple(flat_input_token_ids),
            flat_positions=tuple(flat_positions),
            sampling_params=tuple(sampling_params),
            num_scheduled_tokens=len(flat_input_token_ids),
            num_prefill_reqs=sum(1 for s in scheduled_requests if s.is_prefill),
            num_decode_reqs=sum(1 for s in scheduled_requests if not s.is_prefill),
        )

    def update_from_outputs(
        self,
        scheduler_output: SchedulerOutput,
        sampled_token_ids: Mapping[str, int],
        *,
        stop_token_ids: set[int] | None = None,
    ) -> SchedulerUpdateResult:
        step_outputs: list[RequestStepOutput] = []
        finished_requests: list[FinishedRequestState] = []

        for scheduled_request in scheduler_output.scheduled_requests:
            request = self.requests[scheduled_request.request_id]
            request.num_computed_tokens += scheduled_request.query_len
            sampled_token_id: int | None = None
            finished = False
            finish_reason: FinishReason | None = None

            if scheduled_request.should_sample:
                sampled_token_id = sampled_token_ids[request.request_id]
                request.append_output_token_ids(sampled_token_id)
                if stop_token_ids is not None and sampled_token_id in stop_token_ids:
                    request.mark_finished(RequestStatus.FINISHED_STOPPED)
                    finished = True
                    finish_reason = request.get_finished_reason()
                elif request.num_output_tokens >= request.max_tokens:
                    request.mark_finished(RequestStatus.FINISHED_LENGTH_CAPPED)
                    finished = True
                    finish_reason = request.get_finished_reason()
                if finished:
                    finished_requests.append(
                        FinishedRequestState(
                            request_id=request.request_id,
                            output_token_ids=request.output_token_ids,
                            finish_reason=finish_reason,
                        )
                    )
                    self._finalize_finished_request(request)
                else:
                    request.status = RequestStatus.RUNNING
            else:
                request.status = RequestStatus.RUNNING

            step_outputs.append(
                RequestStepOutput(
                    request_id=request.request_id,
                    sampled_token_id=sampled_token_id,
                    finished=finished,
                    finish_reason=finish_reason,
                )
            )

        return SchedulerUpdateResult(
            step_outputs=tuple(step_outputs),
            finished_requests=tuple(finished_requests),
        )

    def _schedule_request(
        self,
        request: Request,
        *,
        chunk_len: int,
        flat_input_token_ids: list[int],
        flat_positions: list[int],
        sampling_params: list,
        from_waiting: bool = False,
    ) -> ScheduledRequest | None:
        if chunk_len <= 0:
            return None

        original_num_computed_tokens = request.num_computed_tokens
        new_computed_blocks = None
        if from_waiting and request.num_computed_tokens == 0:
            new_computed_blocks, num_computed_tokens = self.kv_manager.get_computed_blocks(request)
            request.num_computed_tokens = num_computed_tokens

        remaining = request.num_tokens - request.num_computed_tokens
        chunk_len = min(chunk_len, remaining)
        if chunk_len <= 0:
            request.num_computed_tokens = original_num_computed_tokens
            return None

        if not self.kv_manager.can_fit(request, chunk_len, new_computed_blocks):
            request.num_computed_tokens = original_num_computed_tokens
            return None
        if self.kv_manager.allocate_slots(request, chunk_len, new_computed_blocks) is None:
            request.num_computed_tokens = original_num_computed_tokens
            return None

        if from_waiting:
            self._remove_request_id(self.waiting, request.request_id)
            self.running.append(request.request_id)
            request.status = RequestStatus.RUNNING

        context_len = request.num_computed_tokens
        flat_start = len(flat_input_token_ids)
        flat_end = flat_start + chunk_len
        is_prefill = request.num_computed_tokens < request.num_prompt_tokens
        should_sample = request.num_computed_tokens + chunk_len == request.num_tokens
        sample_index = flat_end - 1 if should_sample else None

        flat_input_token_ids.extend(
            request.all_token_ids[
                request.num_computed_tokens : request.num_computed_tokens + chunk_len
            ]
        )
        flat_positions.extend(range(context_len, context_len + chunk_len))
        if should_sample:
            sampling_params.append(request.sampling_params)

        return ScheduledRequest(
            request_id=request.request_id,
            is_prefill=is_prefill,
            flat_start=flat_start,
            flat_end=flat_end,
            context_len=context_len,
            query_len=chunk_len,
            sample_index=sample_index,
            should_sample=should_sample,
        )

    def _finalize_finished_request(self, request: Request) -> None:
        self.kv_manager.free(request)
        self._remove_request_id(self.running, request.request_id)
        self.requests.pop(request.request_id, None)
        self.finished_req_ids.add(request.request_id)

    @staticmethod
    def _is_decode_ready(request: Request) -> bool:
        pending = request.num_tokens - request.num_computed_tokens
        return (
            pending > 0
            and request.num_computed_tokens >= request.num_prompt_tokens
        )

    @staticmethod
    def _remove_request_id(request_ids: list[str], request_id: str) -> None:
        try:
            request_ids.remove(request_id)
        except ValueError:
            return


__all__ = [
    "Scheduler",
]
