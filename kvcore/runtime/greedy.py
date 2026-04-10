"""Minimal greedy generation runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvcore.api.config import EngineConfig, GenerationConfig
from kvcore.api.types import GenerationResult, Request
from kvcore.kv import BlockAllocator, RequestKVView
from kvcore.logging import get_logger
from kvcore.runtime.model_runner import SingleRequestModelRunner


@dataclass(slots=True)
class GreedyGenerationRuntime:
    """Execute the minimal prefill + greedy decode path."""

    adapter: Any
    engine_config: EngineConfig
    allocator: BlockAllocator

    def generate(self, request: Request, generation_config: GenerationConfig) -> GenerationResult:
        import torch

        logger = get_logger("runtime.greedy")
        encoded = self.adapter.encode_prompt(request.prompt)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        prompt_token_ids = input_ids[0].tolist()
        prompt_length = len(prompt_token_ids)
        max_new_tokens = generation_config.max_new_tokens or self.engine_config.max_new_tokens
        runner = SingleRequestModelRunner(
            adapter=self.adapter,
            block_size=self.engine_config.block_size,
        )

        kv_view = RequestKVView.from_token_count(
            seq_id=request.request_id,
            total_tokens=prompt_length,
            num_layers=self.adapter.num_hidden_layers,
            block_size=self.engine_config.block_size,
            allocator=self.allocator,
        )
        sequence_state = runner.initialize_sequence_state(
            request_id=request.request_id,
            prompt_token_ids=prompt_token_ids,
            kv_view=kv_view,
        )

        logger.info(
            "request_id=%s prompt_tokens=%s layers=%s block_size=%s",
            request.request_id,
            prompt_length,
            self.adapter.num_hidden_layers,
            self.engine_config.block_size,
        )

        with torch.no_grad():
            prefill_output = runner.run_prefill(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sequence_state=sequence_state,
            )
            logits = prefill_output.logits
            past_key_values = prefill_output.past_key_values

            generated_token_ids: list[int] = []
            finish_reason = "length"
            eos_token_ids = _resolve_eos_token_ids(generation_config, self.adapter.tokenizer)
            next_token_id = _sample_greedy(logits)

            for step in range(max_new_tokens):
                generated_token_ids.append(next_token_id)
                kv_view.update_total_tokens(
                    prompt_length + len(generated_token_ids),
                    self.allocator,
                )
                logger.debug(
                    "request_id=%s decode_step=%s token_id=%s total_tokens=%s kv_blocks=%s",
                    request.request_id,
                    step,
                    next_token_id,
                    kv_view.total_tokens,
                    kv_view.total_block_count,
                )

                if generation_config.stop_on_eos and next_token_id in eos_token_ids:
                    finish_reason = "eos"
                    break

                next_input_ids = torch.tensor([[next_token_id]], device=self.adapter.device)
                decode_output = runner.run_decode_step(
                    input_ids=next_input_ids,
                    past_key_values=past_key_values,
                    sequence_state=sequence_state,
                )
                logits = decode_output.logits
                past_key_values = decode_output.past_key_values
                next_token_id = _sample_greedy(logits)

            sequence_state.generated_token_ids = list(generated_token_ids)

        all_token_ids = prompt_token_ids + generated_token_ids
        text = self.adapter.decode_tokens(
            generated_token_ids,
            skip_special_tokens=generation_config.skip_special_tokens,
        )
        result = GenerationResult(
            text=text,
            token_ids=all_token_ids,
            generated_token_ids=generated_token_ids,
            finish_reason=finish_reason,
            num_prompt_tokens=prompt_length,
            num_generated_tokens=len(generated_token_ids),
            request_id=request.request_id,
            kv_block_count=kv_view.total_block_count,
            kv_total_tokens=kv_view.total_tokens,
            metadata={
                "model_type": self.adapter.model_type,
                "device": self.adapter.device,
                "block_size": self.engine_config.block_size,
            },
        )
        logger.info(
            "request_id=%s finish_reason=%s generated_tokens=%s kv_blocks=%s",
            request.request_id,
            result.finish_reason,
            result.num_generated_tokens,
            result.kv_block_count,
        )
        kv_view.release(self.allocator)
        return result


def _sample_greedy(logits: Any) -> int:
    return int(logits[:, -1, :].argmax(dim=-1).item())


def _resolve_eos_token_ids(generation_config: GenerationConfig, tokenizer: Any) -> tuple[int, ...]:
    if generation_config.normalized_eos_token_ids:
        return generation_config.normalized_eos_token_ids
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return ()
    if isinstance(eos_token_id, int):
        return (eos_token_id,)
    return tuple(int(token_id) for token_id in eos_token_id)
