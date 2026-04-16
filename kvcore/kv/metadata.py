"""Logical KV metadata types for the first implementation stage."""

from __future__ import annotations

from dataclasses import dataclass, field

from kvcore.kv.allocator import BlockAllocator


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

    @classmethod
    def from_token_count(
        cls,
        *,
        seq_id: str,
        total_tokens: int,
        num_layers: int,
        block_size: int,
        allocator: BlockAllocator,
        canonical_tensor_name: str = "hf_past_key_values",
    ) -> RequestKVView:
        if total_tokens < 0:
            raise ValueError("total_tokens must not be negative")
        layer_states: dict[int, LayerKVState] = {}
        canonical_layer_states: dict[int, CanonicalLayerKVState] = {}
        for layer_id in range(num_layers):
            canonical_layer_states[layer_id] = CanonicalLayerKVState(
                layer_id=layer_id,
                ref=CanonicalKVRef(
                    tensor_name=canonical_tensor_name,
                    layer_id=layer_id,
                    block_size_factor=1,
                ),
            )
            blocks = _build_blocks_for_layer(
                allocator=allocator,
                seq_id=seq_id,
                layer_id=layer_id,
                total_tokens=total_tokens,
                block_size=block_size,
            )
            layer_states[layer_id] = LayerKVState(layer_id=layer_id, blocks=blocks)
        return cls(
            seq_id=seq_id,
            total_tokens=total_tokens,
            block_size=block_size,
            layer_states=layer_states,
            canonical_layer_states=canonical_layer_states,
        )

    def update_total_tokens(self, total_tokens: int, allocator: BlockAllocator) -> None:
        if total_tokens < self.total_tokens:
            raise ValueError("total_tokens must not decrease")
        if total_tokens == self.total_tokens:
            return

        old_block_count = _required_block_count(self.total_tokens, self.block_size)
        new_block_count = _required_block_count(total_tokens, self.block_size)
        if new_block_count > old_block_count:
            for layer_id, state in self.layer_states.items():
                for block_index in range(old_block_count, new_block_count):
                    token_start = block_index * self.block_size
                    token_end = min(total_tokens, token_start + self.block_size)
                    state.blocks.append(
                        KVBlock(
                            block_id=allocator.allocate_block_id(),
                            seq_id=self.seq_id,
                            layer_id=layer_id,
                            token_start=token_start,
                            token_end=token_end,
                            capacity=self.block_size,
                        )
                    )

        last_token_end = total_tokens
        for state in self.layer_states.values():
            if state.blocks:
                state.blocks[-1].token_end = last_token_end
        self.total_tokens = total_tokens

    def release(self, allocator: BlockAllocator) -> None:
        block_ids: list[int] = []
        for state in self.layer_states.values():
            for block in state.blocks:
                block.alive = False
                block_ids.append(block.block_id)
            state.blocks.clear()
        allocator.release_many(block_ids)
        self.total_tokens = 0

    @property
    def total_block_count(self) -> int:
        return sum(len(state.blocks) for state in self.layer_states.values())


def _required_block_count(total_tokens: int, block_size: int) -> int:
    if total_tokens == 0:
        return 0
    return (total_tokens + block_size - 1) // block_size


def _build_blocks_for_layer(
    *,
    allocator: BlockAllocator,
    seq_id: str,
    layer_id: int,
    total_tokens: int,
    block_size: int,
) -> list[KVBlock]:
    blocks: list[KVBlock] = []
    for block_index in range(_required_block_count(total_tokens, block_size)):
        token_start = block_index * block_size
        token_end = min(total_tokens, token_start + block_size)
        blocks.append(
            KVBlock(
                block_id=allocator.allocate_block_id(),
                seq_id=seq_id,
                layer_id=layer_id,
                token_start=token_start,
                token_end=token_end,
                capacity=block_size,
            )
        )
    return blocks
