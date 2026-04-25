from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from kvcore.model.kv_runtime import PagedAttentionMetadata


@dataclass(frozen=True, slots=True)
class ForwardContext:
    attn_metadata: PagedAttentionMetadata


_forward_context: ContextVar[ForwardContext | None] = ContextVar(
    "kvcore_forward_context",
    default=None,
)


@contextmanager
def set_forward_context(context: ForwardContext) -> Iterator[ForwardContext]:
    token = _forward_context.set(context)
    try:
        yield context
    finally:
        _forward_context.reset(token)


def get_forward_context() -> ForwardContext:
    context = _forward_context.get()
    if context is None:
        raise RuntimeError(
            "Forward context is not set. ModelRunner must wrap model execution "
            "with set_forward_context before attention is computed."
        )
    return context


__all__ = [
    "ForwardContext",
    "get_forward_context",
    "set_forward_context",
]
