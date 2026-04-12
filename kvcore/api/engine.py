"""User-facing engine entry point."""

from __future__ import annotations

from dataclasses import dataclass

from kvcore.api.config import EngineConfig, GenerationConfig
from kvcore.api.types import GenerationResult, Request
from kvcore.engine import LLMEngine


@dataclass(slots=True)
class Engine:
    """User-facing wrapper around the top-level LLMEngine."""

    config: EngineConfig
    llm_engine: LLMEngine

    @classmethod
    def from_pretrained(cls, config: EngineConfig | None = None) -> Engine:
        llm_engine = LLMEngine.from_pretrained(config)
        return cls(config=llm_engine.config, llm_engine=llm_engine)

    @property
    def adapter(self):
        return self.llm_engine.adapter

    def generate(
        self,
        request: Request,
        generation_config: GenerationConfig | None = None,
    ) -> GenerationResult:
        return self.llm_engine.generate(
            request=request,
            generation_config=generation_config,
        )
