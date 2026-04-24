"""Attention backend abstractions and minimal implementations."""

from kvcore.model.attn_backend.base import AttentionBackend, AttentionType
from kvcore.model.attn_backend.torch_paged import TorchPagedAttentionBackend
from kvcore.model.attn_backend.triton_paged import TritonPagedAttentionBackend


def build_attention_backend(
    backend: str | AttentionBackend | None = None,
) -> AttentionBackend:
    if backend is None or backend == "torch_paged":
        return TorchPagedAttentionBackend()
    if backend == "triton_paged":
        return TritonPagedAttentionBackend()
    if hasattr(backend, "forward"):
        return backend
    raise ValueError(f"Unsupported attention backend: {backend!r}")


__all__ = [
    "AttentionBackend",
    "AttentionType",
    "TorchPagedAttentionBackend",
    "TritonPagedAttentionBackend",
    "build_attention_backend",
]
