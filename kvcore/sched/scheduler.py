from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from kvcore.kv.compression import (
    KVCompressionConfig,
    KVCompressionResult,
    RandomKVBlockCompressor,
)
from kvcore.kv.kv_manager import KVCacheBlocks, KVManager, KVManagerConfig
from kvcore.kv.kv_metrics import KVCacheMetricsCollector
from kvcore.kv.kv_utils import get_request_block_hasher
from kvcore.sched.interface import SchedulerInterface
from kvcore.sched.request_queue import RequestQueue
from kvcore.sched.utils import (
    CachedRequestData,
    FinishedRequestState,
    NewRequestData,
    RequestStepOutput,
    SchedulerConfig,
    SchedulerOutput,
    SchedulerUpdateResult,
)
from kvcore.utils.log import get_logger
from kvcore.utils.request import FinishReason, Request, RequestStatus

logger = get_logger(__name__)


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


class Scheduler(SchedulerInterface):
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
        self.waiting = RequestQueue()
        self.running: list[Request] = []
        self.finished_req_ids: set[str] = set()
        self.request_block_hasher = get_request_block_hasher(
            kv_manager_config.layer_specs[0].block_size
        )
        self.last_no_progress_state: SchedulerNoProgressState | None = None
        logger.info(
            "Scheduler initialized max_num_seqs=%d max_num_scheduled_tokens=%d "
            "max_partial_prefills=%d max_long_partial_prefills=%d",
            self.scheduler_config.max_num_seqs,
            self.scheduler_config.max_num_scheduled_tokens,
            self.scheduler_config.max_num_partial_prefills,
            self.scheduler_config.max_long_partial_prefills,
        )

    def add_request(self, request: Request) -> None:
        if request.request_id in self.requests:
            raise ValueError(f"Duplicate request_id: {request.request_id}")
        if request._block_hasher is None:  # type: ignore[attr-defined]
            request._block_hasher = self.request_block_hasher  # type: ignore[attr-defined]
            request.update_block_hashes()
        self.requests[request.request_id] = request
        self.waiting.add_request(request)
        logger.info(
            "Scheduler queued request request_id=%s prompt_tokens=%d max_tokens=%d",
            request.request_id,
            request.num_prompt_tokens,
            request.max_tokens,
        )

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> tuple[FinishedRequestState, ...]:
        if not RequestStatus.is_finished(finished_status):
            raise ValueError(f"finished_status must be a finished status, got {finished_status}")
        if isinstance(request_ids, str):
            request_id_set = {request_ids}
        elif request_ids is None:
            request_id_set = set(self.requests)
        else:
            request_id_set = set(request_ids)

        finished_requests: list[FinishedRequestState] = []
        waiting_to_remove: list[Request] = []
        running_to_remove: set[str] = set()
        for request_id in request_id_set:
            request = self.requests.get(request_id)
            if request is None or request.is_finished():
                continue
            request.mark_finished(finished_status)
            if request.status == finished_status:
                waiting_to_remove.append(request)
                running_to_remove.add(request.request_id)
            finished_requests.append(self._free_finished_request(request))

        if waiting_to_remove:
            self.waiting.remove_requests(waiting_to_remove)
        if running_to_remove:
            self.running = [
                request
                for request in self.running
                if request.request_id not in running_to_remove
            ]
        return tuple(finished_requests)

    def get_num_unfinished_requests(self) -> int:
        return len(self.running) + len(self.waiting)

    def has_finished_requests(self) -> bool:
        return bool(self.finished_req_ids)

    def get_request_counts(self) -> tuple[int, int]:
        return len(self.running), len(self.waiting)

    def shutdown(self) -> None:
        self.finish_requests(None, RequestStatus.FINISHED_ABORTED)

    def schedule(self) -> SchedulerOutput:
        self.last_no_progress_state = None
        token_budget = self.scheduler_config.max_num_scheduled_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        logger.debug(
            "Scheduler step begin waiting=%d running=%d token_budget=%d",
            len(self.waiting),
            len(self.running),
            token_budget,
        )

        scheduled_new_reqs: list[NewRequestData] = []
        cached_req_ids: list[str] = []
        cached_new_token_ids: list[tuple[int, ...]] = []
        cached_block_ids: list[tuple[tuple[int, ...], ...]] = []
        cached_num_computed_tokens: list[int] = []
        cached_num_output_tokens: list[int] = []
        num_scheduled_tokens: dict[str, int] = {}
        num_partial_prefills = 0
        num_long_partial_prefills = 0

        req_index = 0
        while (
            req_index < len(self.running)
            and len(num_scheduled_tokens) < max_num_seqs
            and token_budget > 0
        ):
            request = self.running[req_index]
            if request.is_finished():
                req_index += 1
                continue
            scheduled = self._try_schedule_request(
                request,
                token_budget=token_budget,
                new_computed_blocks=None,
                scheduled_new_reqs=scheduled_new_reqs,
                cached_req_ids=cached_req_ids,
                cached_new_token_ids=cached_new_token_ids,
                cached_block_ids=cached_block_ids,
                cached_num_computed_tokens=cached_num_computed_tokens,
                cached_num_output_tokens=cached_num_output_tokens,
                num_scheduled_tokens=num_scheduled_tokens,
                num_partial_prefills=num_partial_prefills,
                num_long_partial_prefills=num_long_partial_prefills,
                from_waiting=False,
            )
            if scheduled is None:
                break
            num_tokens, is_prefill = scheduled
            if is_prefill:
                num_partial_prefills += 1
                if self._is_long_prefill(request):
                    num_long_partial_prefills += 1
            token_budget -= num_tokens
            req_index += 1

        skipped_waiting: list[Request] = []
        while (
            self.waiting
            and len(num_scheduled_tokens) < max_num_seqs
            and token_budget > 0
        ):
            request = self.waiting.pop_request()
            original_num_computed_tokens = request.num_computed_tokens
            new_computed_blocks: KVCacheBlocks | None = None
            if request.num_computed_tokens == 0:
                new_computed_blocks, num_computed_tokens = self.kv_manager.get_computed_blocks(
                    request
                )
                request.num_computed_tokens = num_computed_tokens

            scheduled = self._try_schedule_request(
                request,
                token_budget=token_budget,
                new_computed_blocks=new_computed_blocks,
                scheduled_new_reqs=scheduled_new_reqs,
                cached_req_ids=cached_req_ids,
                cached_new_token_ids=cached_new_token_ids,
                cached_block_ids=cached_block_ids,
                cached_num_computed_tokens=cached_num_computed_tokens,
                cached_num_output_tokens=cached_num_output_tokens,
                num_scheduled_tokens=num_scheduled_tokens,
                num_partial_prefills=num_partial_prefills,
                num_long_partial_prefills=num_long_partial_prefills,
                from_waiting=True,
            )
            if scheduled is None:
                request.num_computed_tokens = original_num_computed_tokens
                if self._is_prefill(request) and not self._can_schedule_partial_prefill(
                    request,
                    num_partial_prefills,
                    num_long_partial_prefills,
                ):
                    skipped_waiting.append(request)
                    continue
                self.waiting.prepend_requests((*skipped_waiting, request))
                skipped_waiting = []
                break

            num_tokens, is_prefill = scheduled
            request.status = RequestStatus.RUNNING
            self.running.append(request)
            if is_prefill:
                num_partial_prefills += 1
                if self._is_long_prefill(request):
                    num_long_partial_prefills += 1
            token_budget -= num_tokens

        if skipped_waiting:
            self.waiting.prepend_requests(skipped_waiting)

        finished_req_ids = frozenset(self.finished_req_ids)
        self.finished_req_ids.clear()
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        if total_num_scheduled_tokens == 0 and not finished_req_ids:
            self.last_no_progress_state = SchedulerNoProgressState(
                waiting=tuple(request.request_id for request in self.waiting),
                running=tuple(request.request_id for request in self.running),
                free_blocks=self.kv_manager.block_pool.get_num_free_blocks(),
                max_num_seqs=max_num_seqs,
                max_num_scheduled_tokens=self.scheduler_config.max_num_scheduled_tokens,
            )
            logger.warning(self.last_no_progress_state.format_message())
            return SchedulerOutput.empty()

        new_block_ids_to_zero = tuple(sorted(set(self.kv_manager.take_new_block_ids())))
        logger.info(
            "Scheduler step scheduled reqs=%d tokens=%d new_reqs=%d cached_reqs=%d "
            "finished=%d zero_blocks=%d",
            len(num_scheduled_tokens),
            total_num_scheduled_tokens,
            len(scheduled_new_reqs),
            len(cached_req_ids),
            len(finished_req_ids),
            len(new_block_ids_to_zero),
        )

        return SchedulerOutput(
            scheduled_new_reqs=tuple(scheduled_new_reqs),
            scheduled_cached_reqs=CachedRequestData(
                req_ids=tuple(cached_req_ids),
                new_token_ids=tuple(cached_new_token_ids),
                block_ids=tuple(cached_block_ids),
                num_computed_tokens=tuple(cached_num_computed_tokens),
                num_output_tokens=tuple(cached_num_output_tokens),
            ),
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            finished_req_ids=finished_req_ids,
            new_block_ids_to_zero=new_block_ids_to_zero,
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
        logger.info(
            "Compressing KV cache request_id=%s drop_ratio=%.3f seed=%s max_blocks=%s",
            request_id,
            drop_ratio,
            seed,
            max_blocks,
        )
        # TODO 压缩器应该是一个独立的模块，不应该放在这个函数中
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

        for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[request_id]
            request.num_computed_tokens += num_tokens
            self.kv_manager.cache_blocks(request, request.num_computed_tokens)
            sampled_token_id: int | None = None
            finished = False
            finish_reason: FinishReason | None = None

            if request.num_computed_tokens >= request.num_tokens:
                sampled_token_id = sampled_token_ids[request.request_id]
                request.append_output_token_ids(sampled_token_id)
                logger.debug(
                    "Scheduler consumed sampled token request_id=%s token_id=%d",
                    request.request_id,
                    sampled_token_id,
                )
                if stop_token_ids is not None and sampled_token_id in stop_token_ids:
                    request.mark_finished(RequestStatus.FINISHED_STOPPED)
                    finished = True
                    finish_reason = request.get_finished_reason()
                elif request.num_output_tokens >= request.max_tokens:
                    request.mark_finished(RequestStatus.FINISHED_LENGTH_CAPPED)
                    finished = True
                    finish_reason = request.get_finished_reason()
                if finished:
                    finished_requests.append(self._free_finished_request(request))
                    logger.info(
                        "Request finished request_id=%s reason=%s output_tokens=%d",
                        request.request_id,
                        finish_reason.value if finish_reason is not None else None,
                        request.num_output_tokens,
                    )
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

    def _try_schedule_request(
        self,
        request: Request,
        *,
        token_budget: int,
        new_computed_blocks: KVCacheBlocks | None,
        scheduled_new_reqs: list[NewRequestData],
        cached_req_ids: list[str],
        cached_new_token_ids: list[tuple[int, ...]],
        cached_block_ids: list[tuple[tuple[int, ...], ...]],
        cached_num_computed_tokens: list[int],
        cached_num_output_tokens: list[int],
        num_scheduled_tokens: dict[str, int],
        num_partial_prefills: int,
        num_long_partial_prefills: int,
        from_waiting: bool,
    ) -> tuple[int, bool] | None:
        if token_budget <= 0:
            return None
        is_prefill = self._is_prefill(request)
        if is_prefill and not self._can_schedule_partial_prefill(
            request,
            num_partial_prefills,
            num_long_partial_prefills,
        ):
            logger.debug(
                "Skip request due to partial prefill limits request_id=%s",
                request.request_id,
            )
            return None

        remaining = self._get_num_new_tokens_to_schedule(request)
        if remaining <= 0:
            return None
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < remaining:
            remaining = threshold
        num_tokens = min(remaining, token_budget)
        if num_tokens <= 0:
            return None

        original_num_computed_tokens = request.num_computed_tokens
        if not self.kv_manager.can_fit(request, num_tokens, new_computed_blocks):
            request.num_computed_tokens = original_num_computed_tokens
            logger.debug(
                "Request cannot fit request_id=%s num_tokens=%d free_blocks=%d",
                request.request_id,
                num_tokens,
                self.kv_manager.block_pool.get_num_free_blocks(),
            )
            return None
        allocated_blocks = self.kv_manager.allocate_slots(
            request,
            num_tokens,
            new_computed_blocks,
        )
        if allocated_blocks is None:
            request.num_computed_tokens = original_num_computed_tokens
            logger.debug(
                "KV allocation returned None request_id=%s num_tokens=%d",
                request.request_id,
                num_tokens,
            )
            return None

        context_len = request.num_computed_tokens
        token_ids = request.all_token_ids[context_len : context_len + num_tokens]
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
                )
            )
        else:
            cached_req_ids.append(request.request_id)
            cached_new_token_ids.append(tuple(token_ids))
            cached_block_ids.append(block_ids)
            cached_num_computed_tokens.append(context_len)
            cached_num_output_tokens.append(request.num_output_tokens)

        num_scheduled_tokens[request.request_id] = num_tokens
        logger.debug(
            "Scheduled request request_id=%s prefill=%s context_len=%d num_tokens=%d",
            request.request_id,
            is_prefill,
            context_len,
            num_tokens,
        )
        return num_tokens, is_prefill

    def _free_finished_request(self, request: Request) -> FinishedRequestState:
        self.kv_manager.free(request)
        self.running = [
            running_request
            for running_request in self.running
            if running_request.request_id != request.request_id
        ]
        self.requests.pop(request.request_id, None)
        self.finished_req_ids.add(request.request_id)
        logger.debug("Finalized request request_id=%s", request.request_id)
        return FinishedRequestState(
            request_id=request.request_id,
            output_token_ids=request.output_token_ids,
            finish_reason=request.get_finished_reason(),
        )

    @staticmethod
    def _is_prefill(request: Request) -> bool:
        return request.num_computed_tokens < request.num_prompt_tokens

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


__all__ = [
    "Scheduler",
]
