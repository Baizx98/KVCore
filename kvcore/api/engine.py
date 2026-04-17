"""Public request-facing LLM engine entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvcore.api.config import EngineConfig, GenerationConfig
from kvcore.api.types import GenerationResult, Request
from kvcore.engine import Engine


@dataclass(slots=True)
class LLMEngine:
    """Thin public API wrapper around the internal inference engine."""

    config: EngineConfig
    engine: Engine

    @classmethod
    def from_pretrained(cls, config: EngineConfig | None = None) -> LLMEngine:
        engine = Engine.from_pretrained(config)
        return cls(config=engine.config, engine=engine)

    @property
    def model(self) -> Any:
        return self.engine.model

    def generate(
        self,
        request: Request,
        generation_config: GenerationConfig | None = None,
    ) -> GenerationResult:
        return self.engine.generate(
            request=request,
            generation_config=generation_config,
        )
