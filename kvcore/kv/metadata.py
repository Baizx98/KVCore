"""Logical KV metadata containers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CanonicalKVTensor:
    """Canonical tensor identity exposed to KV management."""

    name: str
    block_size: int


@dataclass(slots=True)
class CanonicalKVRef:
    """Reference from a layer to a canonical KV tensor."""

    tensor_name: str
    layer_id: int
    block_size_factor: int = 1


@dataclass(slots=True)
class KVBlock:
    """Logical block ownership for one layer and one token span."""

    block_id: int
    seq_id: str
    layer_id: int
    token_start: int
    token_end: int
    capacity: int
    residency: str = "gpu"
    alive: bool = True
    block_hash: str | None = None


@dataclass(slots=True)
class LayerKVState:
    """Per-layer logical blocks for a request."""

    layer_id: int
    blocks: list[KVBlock] = field(default_factory=list)


@dataclass(slots=True)
class CanonicalLayerKVState:
    """Canonical layer mapping used by the KV subsystem."""

    layer_id: int
    ref: CanonicalKVRef


@dataclass(slots=True)
class RequestKVView:
    """A request-local logical view over per-layer KV blocks."""

    seq_id: str
    total_tokens: int
    block_size: int
    layer_states: dict[int, LayerKVState]
    canonical_layer_states: dict[int, CanonicalLayerKVState]

    @property
    def total_block_count(self) -> int:
        return sum(len(state.blocks) for state in self.layer_states.values())
