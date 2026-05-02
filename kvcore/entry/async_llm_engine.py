from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from kvcore.config import KVCoreConfig
from kvcore.engine.engine_core import EngineCore
from kvcore.entry.llm_engine import GenerationOutput, GenerationRequest, LLMEngine
from kvcore.utils.log import get_logger
from kvcore.utils.sampling_params import SamplingParams

logger = get_logger(__name__)


class AsyncLLMEngine:
    """Small asyncio front-end over the synchronous EngineCore step loop."""

    def __init__(
        self,
        config: KVCoreConfig,
    ) -> None:
        logger.info("Initializing AsyncLLMEngine")
        self.engine_core = EngineCore(config=config)
        self._pending_queue: asyncio.Queue[GenerationRequest] = asyncio.Queue()
        self._futures: dict[str, asyncio.Future[GenerationOutput]] = {}

    async def generate(
        self,
        requests: Sequence[GenerationRequest | Mapping[str, Any]],
    ) -> list[GenerationOutput]:
        logger.info("Async generate begin requests=%d", len(requests))
        futures = []
        for request in requests:
            normalized_request = LLMEngine._normalize_request(request)
            futures.append(
                await self.submit(
                    normalized_request.request_id,
                    normalized_request.messages,
                    normalized_request.sampling_params,
                )
            )
        await self.run_until_idle()
        outputs = [await future for future in futures]
        logger.info("Async generate done outputs=%d", len(outputs))
        return outputs

    async def submit(
        self,
        request_id: str,
        messages: Sequence[Mapping[str, Any]],
        sampling_params: SamplingParams,
    ) -> asyncio.Future[GenerationOutput]:
        if request_id in self._futures:
            raise ValueError(f"Duplicate request_id: {request_id}")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[GenerationOutput] = loop.create_future()
        self._futures[request_id] = future
        await self._pending_queue.put(
            GenerationRequest(
                request_id=request_id,
                messages=tuple(messages),
                sampling_params=sampling_params,
            )
        )
        logger.info("Async submitted request request_id=%s", request_id)
        return future

    async def run_until_idle(self) -> None:
        logger.debug("Async run_until_idle begin")
        while not self._pending_queue.empty() or self.engine_core.has_unfinished_requests():
            await self._drain_pending_queue()
            if not self.engine_core.has_unfinished_requests():
                await asyncio.sleep(0)
                continue
            step_output = self.engine_core.step()
            self._resolve_finished_futures()
            if not step_output.request_outputs and self.engine_core.has_unfinished_requests():
                no_progress_state = self.engine_core.scheduler.last_no_progress_state
                if no_progress_state is not None:
                    logger.error(no_progress_state.format_message())
                    raise RuntimeError(no_progress_state.format_message())
                logger.error("Async engine made no scheduling progress")
                raise RuntimeError(
                    "Engine made no scheduling progress while requests are still unfinished."
                )
            await asyncio.sleep(0)
        logger.debug("Async run_until_idle done")

    async def _drain_pending_queue(self) -> None:
        while not self._pending_queue.empty():
            request = await self._pending_queue.get()
            self.engine_core.add_request(
                request_id=request.request_id,
                messages=request.messages,
                sampling_params=request.sampling_params,
            )
            logger.debug("Async drained request request_id=%s", request.request_id)

    def _resolve_finished_futures(self) -> None:
        for request_id, future in list(self._futures.items()):
            if future.done():
                self._futures.pop(request_id, None)
                continue
            finished_output = self.engine_core.take_finished_output(request_id)
            if finished_output is None:
                continue
            future.set_result(
                GenerationOutput(
                    request_id=request_id,
                    output_text=self.engine_core.tokenizer_manager.decode(
                        finished_output.output_token_ids
                    ),
                    output_token_ids=finished_output.output_token_ids,
                    finish_reason=(
                        None
                        if finished_output.finish_reason is None
                        else finished_output.finish_reason.value
                    ),
                )
            )
            self._futures.pop(request_id, None)
            logger.info("Async resolved request request_id=%s", request_id)


__all__ = [
    "AsyncLLMEngine",
]
