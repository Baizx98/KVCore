from __future__ import annotations

from kvcore.entry.llm_engine import LLMEngine


class AsyncLLMEngine:
    """Minimal synchronous shim kept for future async expansion."""

    def __init__(self, *args, **kwargs) -> None:
        self._sync_engine = LLMEngine(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._sync_engine, name)


__all__ = [
    "AsyncLLMEngine",
]
