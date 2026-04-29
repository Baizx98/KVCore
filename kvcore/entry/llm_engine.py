from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kvcore.config import KVCoreConfig
from kvcore.engine.engine_core import EngineConfig, EngineCore
from kvcore.model.model_loader import ModelLoadConfig
from kvcore.utils.log import get_logger
from kvcore.utils.sampling_params import SamplingParams

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    request_id: str
    messages: tuple[Mapping[str, Any], ...]
    sampling_params: SamplingParams


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    request_id: str
    output_text: str
    output_token_ids: tuple[int, ...]
    finish_reason: str | None


class LLMEngine:
    def __init__(
        self,
        load_config: ModelLoadConfig | KVCoreConfig | None = None,
        engine_config: EngineConfig | None = None,
        *,
        config: KVCoreConfig | None = None,
        **core_kwargs,
    ) -> None:
        logger.info("Initializing LLMEngine")
        self.engine_core = EngineCore(
            load_config=load_config,
            engine_config=engine_config,
            config=config,
            **core_kwargs,
        )

    def add_request(
        self,
        request_id: str,
        messages: Sequence[Mapping[str, Any]],
        sampling_params: SamplingParams,
    ) -> None:
        self.engine_core.add_request(
            request_id=request_id,
            messages=messages,
            sampling_params=sampling_params,
        )

    def step(self):
        return self.engine_core.step()

    def generate(
        self,
        requests: Sequence[GenerationRequest | Mapping[str, Any]],
    ) -> list[GenerationOutput]:
        normalized_requests = [self._normalize_request(request) for request in requests]
        logger.info("LLMEngine generate begin requests=%d", len(normalized_requests))
        for request in normalized_requests:
            self.add_request(
                request_id=request.request_id,
                messages=request.messages,
                sampling_params=request.sampling_params,
            )

        while self.engine_core.has_unfinished_requests():
            step_output = self.step()
            if not step_output.request_outputs and self.engine_core.has_unfinished_requests():
                logger.error("LLMEngine generate made no progress")
                raise RuntimeError(
                    "Engine made no scheduling progress while requests are still unfinished. "
                    "This usually means the KV block budget is too small."
                )

        outputs: list[GenerationOutput] = []
        for request in normalized_requests:
            finished_output = self.engine_core.take_finished_output(request.request_id)
            if finished_output is None:
                continue
            outputs.append(
                GenerationOutput(
                    request_id=request.request_id,
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
        logger.info("LLMEngine generate done outputs=%d", len(outputs))
        return outputs

    @staticmethod
    def _normalize_request(
        request: GenerationRequest | Mapping[str, Any],
    ) -> GenerationRequest:
        if isinstance(request, GenerationRequest):
            return request
        return GenerationRequest(
            request_id=request["request_id"],
            messages=tuple(request["messages"]),
            sampling_params=request["sampling_params"],
        )


__all__ = [
    "GenerationOutput",
    "GenerationRequest",
    "LLMEngine",
]
