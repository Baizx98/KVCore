"""Minimal scheduler with waiting and running queues."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from kvcore.api.config import GenerationConfig
from kvcore.api.types import GenerationResult, Request
from kvcore.logging import get_logger
from kvcore.scheduler.state import RequestState, ScheduledBatch


@dataclass(slots=True)
class Scheduler:
    """Maintain waiting and running queues and emit explicit prefill/decode batches."""

    waiting: deque[RequestState] = field(default_factory=deque)
    running: deque[RequestState] = field(default_factory=deque)

    def add_request(self, request: Request) -> None:
        self.waiting.append(RequestState(request=request))

    def has_pending_requests(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> ScheduledBatch | None:
        if self.waiting:
            request_state = self.waiting.popleft()
            request_state.status = "running"
            self.running.append(request_state)
            return self._build_batch(mode="prefill", request_state=request_state)

        if self.running:
            request_state = self.running[0]
            if request_state.finished:
                self.running.popleft()
                return self.schedule()
            return self._build_batch(mode="decode", request_state=request_state)

        return None

    def commit_step(
        self,
        *,
        scheduled_batch: ScheduledBatch,
        generation_config: GenerationConfig,
        step_output: Any,
        model_runner: Any,
        kv_manager: Any,
        default_max_new_tokens: int,
    ) -> GenerationResult | None:
        logger = get_logger("scheduler")
        request_state = scheduled_batch.request_states[0]
        next_token_id = int(step_output.logits[:, -1, :].argmax(dim=-1).item())

        if scheduled_batch.mode == "prefill":
            request_state.past_key_values = step_output.past_key_values
            request_state.sequence_state = step_output.batch_context.sequence_state

        request_state.generated_token_ids.append(next_token_id)
        if request_state.sequence_state is not None:
            request_state.sequence_state.generated_token_ids.append(next_token_id)
        request_state.past_key_values = step_output.past_key_values

        eos_token_ids = _resolve_eos_token_ids(generation_config, model_runner.adapter.tokenizer)
        max_new_tokens = generation_config.max_new_tokens or default_max_new_tokens
        if generation_config.stop_on_eos and next_token_id in eos_token_ids:
            request_state.finished = True
            request_state.finish_reason = "eos"
        elif len(request_state.generated_token_ids) >= max_new_tokens:
            request_state.finished = True
            request_state.finish_reason = "length"

        logger.debug(
            "request_id=%s mode=%s generated=%s finished=%s",
            request_state.request_id,
            scheduled_batch.mode,
            len(request_state.generated_token_ids),
            request_state.finished,
        )

        if not request_state.finished:
            return None

        sequence_state = request_state.sequence_state
        if sequence_state is None:
            raise RuntimeError(f"request_id={request_state.request_id!r} has no sequence state")
        token_ids = request_state.prompt_token_ids + request_state.generated_token_ids
        kv_manager.cache_request_blocks(
            request_id=request_state.request_id,
            token_ids=token_ids,
        )
        result = GenerationResult(
            text=model_runner.adapter.decode_tokens(
                request_state.generated_token_ids,
                skip_special_tokens=generation_config.skip_special_tokens,
            ),
            token_ids=token_ids,
            generated_token_ids=list(request_state.generated_token_ids),
            finish_reason=request_state.finish_reason or "length",
            num_prompt_tokens=len(request_state.prompt_token_ids),
            num_generated_tokens=len(request_state.generated_token_ids),
            request_id=request_state.request_id,
            kv_block_count=sequence_state.kv_view.total_block_count,
            kv_total_tokens=sequence_state.kv_view.total_tokens,
            metadata={
                "model_type": model_runner.adapter.model_type,
                "device": model_runner.adapter.device,
                "block_size": kv_manager.block_size,
                "scheduled_mode": scheduled_batch.mode,
            },
        )
        kv_manager.release_request(request_state.request_id)
        if self.running and self.running[0] is request_state:
            self.running.popleft()
        return result

    def _build_batch(self, *, mode: str, request_state: RequestState) -> ScheduledBatch:
        if mode == "prefill":
            flat_input_ids = list(request_state.prompt_token_ids)
            flat_position_ids = list(range(len(flat_input_ids)))
            encoded_inputs = request_state.encoded_inputs
        else:
            last_token_id = request_state.generated_token_ids[-1]
            flat_input_ids = [last_token_id]
            flat_position_ids = [request_state.total_tokens - 1]
            encoded_inputs = None

        return ScheduledBatch(
            mode=mode,
            request_states=[request_state],
            request_ids=[request_state.request_id],
            num_requests=1,
            num_tokens=len(flat_input_ids),
            flat_input_ids=flat_input_ids,
            flat_position_ids=flat_position_ids,
            request_offsets=[0],
            request_token_counts=[len(flat_input_ids)],
            encoded_inputs=encoded_inputs,
        )


def _resolve_eos_token_ids(generation_config: GenerationConfig, tokenizer: Any) -> tuple[int, ...]:
    if generation_config.normalized_eos_token_ids:
        return generation_config.normalized_eos_token_ids
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return ()
    if isinstance(eos_token_id, int):
        return (eos_token_id,)
    return tuple(int(token_id) for token_id in eos_token_id)
