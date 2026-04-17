"""Model runner that bridges scheduled batches to model execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kvcore.logging import get_logger
from kvcore.model_runner.context import StepOutput
from kvcore.model_runner.layer_runner import LayerwiseModelRunner


@dataclass(slots=True)
class ModelRunner:
    """Prepare model inputs, initialize KV cache resources, and execute batches."""

    model: Any
    block_size: int
    kv_manager: Any
    layer_runner: LayerwiseModelRunner = field(init=False)

    def __post_init__(self) -> None:
        self.layer_runner = LayerwiseModelRunner(
            model=self.model,
            block_size=self.block_size,
        )

    @property
    def adapter(self) -> Any:
        return self.model

    def profile_run(self) -> dict[str, int | str]:
        return {
            "num_layers": self.model.num_hidden_layers,
            "block_size": self.block_size,
            "device": self.model.device,
        }

    def run_batch(self, *, scheduled_batch: Any) -> StepOutput:
        logger = get_logger("model_runner")
        request_state = scheduled_batch.request_states[0]

        if scheduled_batch.mode == "prefill":
            encoded = self.model.encode_prompt(request_state.prompt)
            request_state.encoded_inputs = encoded
            request_state.prompt_token_ids = encoded["input_ids"][0].tolist()
            kv_view = self.kv_manager.register_request(
                request_id=request_state.request_id,
                total_tokens=len(request_state.prompt_token_ids),
            )
            sequence_state = self.layer_runner.initialize_sequence_state(
                request_id=request_state.request_id,
                prompt_token_ids=request_state.prompt_token_ids,
                kv_view=kv_view,
            )
            logger.info(
                "request_id=%s prompt_tokens=%s layers=%s block_size=%s",
                request_state.request_id,
                len(request_state.prompt_token_ids),
                self.model.num_hidden_layers,
                self.block_size,
            )
            return self.layer_runner.run_prefill(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                sequence_state=sequence_state,
            )

        next_input_ids = self.model.make_input_ids(
            scheduled_batch.flat_input_ids,
            device=self.model.device,
        )
        self.kv_manager.advance_request(
            request_id=request_state.request_id,
            total_tokens=request_state.total_tokens + 1,
        )
        request_state.sequence_state.kv_view = self.kv_manager.get_request_view(
            request_state.request_id
        )
        return self.layer_runner.run_decode_step(
            input_ids=next_input_ids,
            past_key_values=request_state.past_key_values,
            sequence_state=request_state.sequence_state,
        )
