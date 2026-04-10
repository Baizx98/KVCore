"""User-facing engine entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kvcore.api.config import EngineConfig, GenerationConfig
from kvcore.api.types import GenerationResult, Request
from kvcore.kv import BlockAllocator
from kvcore.logging import setup_logging
from kvcore.model import LlamaModelAdapter
from kvcore.runtime import GreedyGenerationRuntime


@dataclass(slots=True)
class Engine:
    """Minimal engine for the first implementation stage."""

    config: EngineConfig
    adapter: Any
    allocator: BlockAllocator = field(default_factory=BlockAllocator)

    @classmethod
    def from_pretrained(cls, config: EngineConfig | None = None) -> Engine:
        resolved_config = config or EngineConfig()
        setup_logging(resolved_config.log_level)
        adapter = LlamaModelAdapter.from_config(resolved_config)
        return cls(config=resolved_config, adapter=adapter)

    def generate(
        self,
        request: Request,
        generation_config: GenerationConfig | None = None,
    ) -> GenerationResult:
        runtime = GreedyGenerationRuntime(
            adapter=self.adapter,
            engine_config=self.config,
            allocator=self.allocator,
        )
        resolved_generation_config = generation_config or GenerationConfig()
        return runtime.generate(
            request=request,
            generation_config=resolved_generation_config,
        )
