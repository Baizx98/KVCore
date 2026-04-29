from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kvcore.kv.compression import (
    KVCompressionConfig,
    KVCompressionResult,
    RandomKVBlockCompressor,
)
from kvcore.kv.kv_manager import KVManager, KVManagerConfig
from kvcore.kv.kv_metrics import KVCacheMetricsCollector
from kvcore.kv.kv_utils import get_request_block_hasher
from kvcore.sched.utils import (
    CachedRequestData,
    FinishedRequestState,
    NewRequestData,
    RequestStepOutput,
    ScheduledRequest,
    SchedulerConfig,
    SchedulerOutput,
    SchedulerUpdateResult,
)
from kvcore.utils.request import FinishReason, Request, RequestStatus


@dataclass(frozen=True, slots=True)
class SchedulerNoProgressState:
    waiting: tuple[str, ...]
    running: tuple[str, ...]
    free_blocks: int
    max_num_seqs: int
    max_num_scheduled_tokens: int

    def format_message(self) -> str:
        return (
            "Scheduler made no progress "
            f"(waiting={self.waiting}, running={self.running}, "
            f"free_blocks={self.free_blocks}, max_num_seqs={self.max_num_seqs}, "
            f"max_num_scheduled_tokens={self.max_num_scheduled_tokens})"
        )


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
        self.request_block_hasher = get_request_block_hasher(
            kv_manager_config.layer_specs[0].block_size
        )
        self.last_no_progress_state: SchedulerNoProgressState | None = None

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
        return request

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        self.last_no_progress_state = None
        budget = self.scheduler_config.max_num_scheduled_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        scheduled_requests: list[ScheduledRequest] = []
        scheduled_new_reqs: list[NewRequestData] = []
        cached_req_ids: list[str] = []
        cached_new_token_ids: list[tuple[int, ...]] = []
        cached_block_ids: list[tuple[tuple[int, ...], ...]] = []
        cached_num_computed_tokens: list[int] = []
        cached_num_scheduled_tokens: list[int] = []
        cached_is_prefill: list[bool] = []
        cached_should_sample: list[bool] = []
        cached_sample_indices: list[int | None] = []
        total_num_scheduled_tokens = 0
        num_partial_prefills = 0
        num_long_partial_prefills = 0

        # vLLM-style catch-up: schedule each request until its computed tokens
        # catch up to the current token length. KVCore keeps a simpler ordering:
        # already-running requests first, then waiting requests.
        for request_id in list(self.running) + list(self.waiting):
            if len(scheduled_requests) >= max_num_seqs or budget <= 0:
                break
            request = self.requests[request_id]
            from_waiting = request_id in self.waiting
            is_prefill = request.num_computed_tokens < request.num_prompt_tokens
            if is_prefill and not self._can_schedule_partial_prefill(
                request,
                num_partial_prefills,
                num_long_partial_prefills,
            ):
                continue
            original_num_computed_tokens = request.num_computed_tokens
            if from_waiting and request.num_computed_tokens == 0:
                new_computed_blocks, num_computed_tokens = self.kv_manager.get_computed_blocks(
                    request
                )
                request.num_computed_tokens = num_computed_tokens
            else:
                new_computed_blocks = None
            remaining = self._get_num_new_tokens_to_schedule(request)
            if remaining <= 0:
                continue
            chunk_len = min(remaining, budget)
            if from_waiting and self.scheduler_config.reserve_full_prompt_blocks:
                if not self.kv_manager.can_fit(request, remaining):
                    continue
            scheduled = self._schedule_request(
                request,
                chunk_len=chunk_len,
                new_computed_blocks=new_computed_blocks,
                scheduled_token_offset=total_num_scheduled_tokens,
                scheduled_new_reqs=scheduled_new_reqs,
                cached_req_ids=cached_req_ids,
                cached_new_token_ids=cached_new_token_ids,
                cached_block_ids=cached_block_ids,
                cached_num_computed_tokens=cached_num_computed_tokens,
                cached_num_scheduled_tokens=cached_num_scheduled_tokens,
                cached_is_prefill=cached_is_prefill,
                cached_should_sample=cached_should_sample,
                cached_sample_indices=cached_sample_indices,
                from_waiting=from_waiting,
            )
            if scheduled is None:
                request.num_computed_tokens = original_num_computed_tokens
                break
            scheduled_requests.append(scheduled)
            if scheduled.is_prefill:
                num_partial_prefills += 1
                if self._is_long_prefill(request):
                    num_long_partial_prefills += 1
            budget -= scheduled.query_len
            total_num_scheduled_tokens += scheduled.query_len

        if not scheduled_requests:
            self.last_no_progress_state = SchedulerNoProgressState(
                waiting=tuple(self.waiting),
                running=tuple(self.running),
                free_blocks=self.kv_manager.block_pool.get_num_free_blocks(),
                max_num_seqs=max_num_seqs,
                max_num_scheduled_tokens=self.scheduler_config.max_num_scheduled_tokens,
            )
            return SchedulerOutput.empty()

        new_block_ids_to_zero = tuple(sorted(set(self.kv_manager.take_new_block_ids())))

        return SchedulerOutput(
            scheduled_requests=tuple(scheduled_requests),
            num_scheduled_tokens={
                scheduled_request.request_id: scheduled_request.query_len
                for scheduled_request in scheduled_requests
            },
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            num_prefill_reqs=sum(1 for s in scheduled_requests if s.is_prefill),
            num_decode_reqs=sum(1 for s in scheduled_requests if not s.is_prefill),
            new_block_ids_to_zero=new_block_ids_to_zero,
            scheduled_new_reqs=tuple(scheduled_new_reqs),
            scheduled_cached_reqs=CachedRequestData(
                req_ids=tuple(cached_req_ids),
                new_token_ids=tuple(cached_new_token_ids),
                block_ids=tuple(cached_block_ids),
                num_computed_tokens=tuple(cached_num_computed_tokens),
                num_scheduled_tokens=tuple(cached_num_scheduled_tokens),
                is_prefill=tuple(cached_is_prefill),
                should_sample=tuple(cached_should_sample),
                sample_indices=tuple(cached_sample_indices),
            ),
        )

    def compress_kv_cache(
        self,
        request_id: str,
        drop_ratio: float,
        *,
        seed: int | None = None,
        layer_ids: tuple[int, ...] | None = None,
        max_blocks: int | None = None,
    ) -> KVCompressionResult:
        if request_id not in self.requests:
            raise KeyError(f"Unknown request_id: {request_id}")
        compressor = RandomKVBlockCompressor(
            KVCompressionConfig(
                drop_ratio=drop_ratio,
                seed=seed,
                layer_ids=layer_ids,
                max_blocks=max_blocks,
            )
        )
        return compressor.compress(self.kv_manager, (request_id,))

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
        new_computed_blocks,
        scheduled_token_offset: int,
        scheduled_new_reqs: list[NewRequestData],
        cached_req_ids: list[str],
        cached_new_token_ids: list[tuple[int, ...]],
        cached_block_ids: list[tuple[tuple[int, ...], ...]],
        cached_num_computed_tokens: list[int],
        cached_num_scheduled_tokens: list[int],
        cached_is_prefill: list[bool],
        cached_should_sample: list[bool],
        cached_sample_indices: list[int | None],
        from_waiting: bool = False,
    ) -> ScheduledRequest | None:
        if chunk_len <= 0:
            return None

        original_num_computed_tokens = request.num_computed_tokens
        remaining = self._get_num_new_tokens_to_schedule(request)
        chunk_len = min(chunk_len, remaining)
        if chunk_len <= 0:
            request.num_computed_tokens = original_num_computed_tokens
            return None

        if not self.kv_manager.can_fit(request, chunk_len, new_computed_blocks):
            request.num_computed_tokens = original_num_computed_tokens
            return None
        allocated_blocks = self.kv_manager.allocate_slots(
            request,
            chunk_len,
            new_computed_blocks,
        )
        if allocated_blocks is None:
            request.num_computed_tokens = original_num_computed_tokens
            return None

        if from_waiting:
            self._remove_request_id(self.waiting, request.request_id)
            self.running.append(request.request_id)
            request.status = RequestStatus.RUNNING

        context_len = request.num_computed_tokens
        is_prefill = request.num_computed_tokens < request.num_prompt_tokens
        should_sample = request.num_computed_tokens + chunk_len == request.num_tokens
        sample_index = scheduled_token_offset + chunk_len - 1 if should_sample else None
        token_ids = request.all_token_ids[
            request.num_computed_tokens : request.num_computed_tokens + chunk_len
        ]

        block_ids = tuple(
            tuple(layer_block_ids)
            for layer_block_ids in self.kv_manager.get_block_ids(request.request_id)
        )
        if from_waiting:
            scheduled_new_reqs.append(
                NewRequestData(
                    req_id=request.request_id,
                    prompt_token_ids=tuple(request.prompt_token_ids),
                    sampling_params=request.sampling_params,
                    block_ids=block_ids,
                    num_computed_tokens=context_len,
                    num_scheduled_tokens=chunk_len,
                    is_prefill=is_prefill,
                    should_sample=should_sample,
                    sample_index=sample_index,
                )
            )
        else:
            cached_req_ids.append(request.request_id)
            cached_new_token_ids.append(tuple(token_ids))
            cached_block_ids.append(block_ids)
            cached_num_computed_tokens.append(context_len)
            cached_num_scheduled_tokens.append(chunk_len)
            cached_is_prefill.append(is_prefill)
            cached_should_sample.append(should_sample)
            cached_sample_indices.append(sample_index)

        return ScheduledRequest(
            request_id=request.request_id,
            is_prefill=is_prefill,
            context_len=context_len,
            query_len=chunk_len,
            sample_index=sample_index,
            should_sample=should_sample,
            num_computed_tokens=context_len,
            block_ids=block_ids,
        )

    def _finalize_finished_request(self, request: Request) -> None:
        self.kv_manager.free(request)
        self._remove_request_id(self.running, request.request_id)
        self.requests.pop(request.request_id, None)

    @staticmethod
    def _is_decode_ready(request: Request) -> bool:
        pending = request.num_tokens - request.num_computed_tokens
        return (
            pending > 0
            and request.num_computed_tokens >= request.num_prompt_tokens
        )

    @staticmethod
    def _get_num_new_tokens_to_schedule(request: Request) -> int:
        return request.num_tokens - request.num_computed_tokens

    def _can_schedule_partial_prefill(
        self,
        request: Request,
        num_partial_prefills: int,
        num_long_partial_prefills: int,
    ) -> bool:
        if num_partial_prefills >= self.scheduler_config.max_num_partial_prefills:
            return False
        return not (
            self._is_long_prefill(request)
            and num_long_partial_prefills >= self.scheduler_config.max_long_partial_prefills
        )

    def _is_long_prefill(self, request: Request) -> bool:
        threshold = self.scheduler_config.long_prefill_token_threshold
        return threshold > 0 and request.num_prompt_tokens >= threshold

    @staticmethod
    def _remove_request_id(request_ids: list[str], request_id: str) -> None:
        try:
            request_ids.remove(request_id)
        except ValueError:
            return


__all__ = [
    "Scheduler",
]
