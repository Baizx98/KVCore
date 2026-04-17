"""Internal inference engine that coordinates scheduling and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kvcore.api.config import EngineConfig, GenerationConfig
from kvcore.api.types import GenerationResult, Request
from kvcore.kv import KVManager
from kvcore.logging import get_logger, setup_logging
from kvcore.model import load_model_from_config
from kvcore.model_runner import ModelRunner
from kvcore.scheduler import Scheduler


@dataclass(slots=True)
class Engine:
    """Coordinate request scheduling, KV management, and model execution."""

    config: EngineConfig
    model_runner: ModelRunner
    kv_manager: KVManager
    scheduler: Scheduler = field(default_factory=Scheduler)

    @classmethod
    def from_pretrained(cls, config: EngineConfig | None = None) -> Engine:
        resolved_config = config or EngineConfig()
        setup_logging(resolved_config.log_level)
        model = load_model_from_config(resolved_config)
        kv_manager = KVManager.from_model_config(
            num_layers=model.num_hidden_layers,
            block_size=resolved_config.block_size,
            device=model.device,
        )
        model_runner = ModelRunner(
            model=model,
            block_size=resolved_config.block_size,
            kv_manager=kv_manager,
        )
        return cls(
            config=resolved_config,
            model_runner=model_runner,
            kv_manager=kv_manager,
        )

    @property
    def model(self) -> Any:
        return self.model_runner.model

    @property
    def adapter(self) -> Any:
        return self.model

    def generate(
        self,
        request: Request,
        generation_config: GenerationConfig | None = None,
    ) -> GenerationResult:
        logger = get_logger("engine")
        resolved_generation_config = generation_config or GenerationConfig()
        self.scheduler.add_request(request)
        logger.info("request_id=%s submitted to waiting queue", request.request_id)

        result: GenerationResult | None = None
        while self.scheduler.has_pending_requests():
            scheduled_batch = self.scheduler.schedule()
            if scheduled_batch is None:
                break

            logger.debug(
                "scheduled mode=%s requests=%s tokens=%s",
                scheduled_batch.mode,
                scheduled_batch.request_ids,
                scheduled_batch.num_tokens,
            )
            step_output = self.model_runner.run_batch(scheduled_batch=scheduled_batch)
            result = self.scheduler.commit_step(
                scheduled_batch=scheduled_batch,
                generation_config=resolved_generation_config,
                step_output=step_output,
                model_runner=self.model_runner,
                kv_manager=self.kv_manager,
                default_max_new_tokens=self.config.max_new_tokens,
            )
            if result is not None:
                break

        if result is None:
            raise RuntimeError(f"request_id={request.request_id!r} did not produce a result")
        return result
