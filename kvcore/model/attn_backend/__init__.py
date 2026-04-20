"""Attention backend abstractions and minimal implementations."""

from kvcore.model.attn_backend.base import AttentionBackend, AttentionType
from kvcore.model.attn_backend.torch_sdpa import TorchSDPAAttentionBackend


def build_attention_backend(
    backend: str | AttentionBackend | None = None,
) -> AttentionBackend:
    if backend is None or backend == "torch_sdpa":
        return TorchSDPAAttentionBackend()
    if hasattr(backend, "forward"):
        return backend
    raise ValueError(f"Unsupported attention backend: {backend!r}")


__all__ = [
    "AttentionBackend",
    "AttentionType",
    "TorchSDPAAttentionBackend",
    "build_attention_backend",
]
