"""Embedding and LM head layers."""

from __future__ import annotations

from torch import nn


class VocabEmbedding(nn.Embedding):
    """Embedding layer matching decoder-only HF parameter names."""


class ParallelLMHead(nn.Linear):
    """Single-device LM head mirroring vLLM's top-level naming."""
