"""Per-layer single-type KV cache manager."""

from __future__ import annotations

from dataclasses import dataclass, field

from kvcore.kv.block_pool import BlockPool, KVCacheBlock, hash_block_tokens
from kvcore.kv.metadata import KVBlock, LayerKVState


@dataclass(slots=True)
class SingleTypeKVManager:
    """Manage KV blocks for exactly one homogeneous attention layer."""

    layer_id: int
    block_size: int
    block_pool: BlockPool
    request_blocks: dict[str, list[KVBlock]] = field(default_factory=dict)
    num_cached_blocks: dict[str, int] = field(default_factory=dict)

    def register_request(self, *, request_id: str, total_tokens: int) -> LayerKVState:
        if request_id in self.request_blocks:
            raise ValueError(f"request_id={request_id!r} is already registered")
        self.request_blocks[request_id] = []
        self.num_cached_blocks[request_id] = 0
        self.allocate_new_blocks(request_id=request_id, total_tokens=total_tokens)
        return self.get_layer_state(request_id)

    def allocate_new_blocks(self, *, request_id: str, total_tokens: int) -> list[KVBlock]:
        if total_tokens < 0:
            raise ValueError("total_tokens must not be negative")
        blocks = self.request_blocks.setdefault(request_id, [])
        old_block_count = len(blocks)
        new_block_count = _required_block_count(total_tokens, self.block_size)
        if new_block_count <= old_block_count:
            self._update_tail_token_end(request_id=request_id, total_tokens=total_tokens)
            return []

        physical_blocks = self.block_pool.get_new_blocks(new_block_count - old_block_count)
        new_blocks: list[KVBlock] = []
        for offset, physical_block in enumerate(physical_blocks):
            block_index = old_block_count + offset
            token_start = block_index * self.block_size
            kv_block = _logical_block_from_physical(
                physical_block=physical_block,
                request_id=request_id,
                layer_id=self.layer_id,
                token_start=token_start,
                token_end=min(total_tokens, token_start + self.block_size),
                block_size=self.block_size,
            )
            blocks.append(kv_block)
            new_blocks.append(kv_block)
        self._update_tail_token_end(request_id=request_id, total_tokens=total_tokens)
        return new_blocks

    def allocate_new_computed_blocks(
        self,
        *,
        request_id: str,
        physical_blocks: list[KVCacheBlock],
        total_tokens: int,
    ) -> LayerKVState:
        self.block_pool.touch(physical_blocks)
        blocks = self.request_blocks.setdefault(request_id, [])
        old_block_count = len(blocks)
        for offset, physical_block in enumerate(physical_blocks):
            block_index = old_block_count + offset
            token_start = block_index * self.block_size
            blocks.append(
                _logical_block_from_physical(
                    physical_block=physical_block,
                    request_id=request_id,
                    layer_id=self.layer_id,
                    token_start=token_start,
                    token_end=min(total_tokens, token_start + self.block_size),
                    block_size=self.block_size,
                )
            )
        self._update_tail_token_end(request_id=request_id, total_tokens=total_tokens)
        return self.get_layer_state(request_id)

    def advance_request(self, *, request_id: str, total_tokens: int) -> LayerKVState:
        current_tokens = self.total_tokens(request_id)
        if total_tokens < current_tokens:
            raise ValueError("total_tokens must not decrease")
        self.allocate_new_blocks(request_id=request_id, total_tokens=total_tokens)
        return self.get_layer_state(request_id)

    def cache_blocks(
        self,
        *,
        request_id: str,
        token_ids: list[int],
        extra_keys: tuple[object, ...] = (),
    ) -> None:
        blocks = self.request_blocks[request_id]
        num_full_blocks = len(token_ids) // self.block_size
        start_index = self.num_cached_blocks.get(request_id, 0)
        if num_full_blocks <= start_index:
            return

        block_hashes: list[str] = []
        parent_hash: str | None = None
        for block_index in range(num_full_blocks):
            block = blocks[block_index]
            if block.block_hash is not None:
                parent_hash = block.block_hash
                if block_index >= start_index:
                    block_hashes.append(block.block_hash)
                continue
            block_token_ids = token_ids[block.token_start : block.token_start + self.block_size]
            block.block_hash = hash_block_tokens(
                token_ids=block_token_ids,
                parent_hash=parent_hash,
                extra_keys=extra_keys,
            )
            parent_hash = block.block_hash
            if block_index >= start_index:
                block_hashes.append(block.block_hash)

        physical_blocks = [
            self.block_pool.blocks[block.block_id] for block in blocks[start_index:num_full_blocks]
        ]
        self.block_pool.cache_full_blocks(blocks=physical_blocks, block_hashes=block_hashes)
        self.num_cached_blocks[request_id] = num_full_blocks

    def find_longest_cache_hit(
        self,
        *,
        token_ids: list[int],
        max_length: int,
        extra_keys: tuple[object, ...] = (),
    ) -> int:
        num_candidate_blocks = min(max_length, len(token_ids)) // self.block_size
        block_hashes: list[str] = []
        parent_hash: str | None = None
        for block_index in range(num_candidate_blocks):
            start = block_index * self.block_size
            parent_hash = hash_block_tokens(
                token_ids=token_ids[start : start + self.block_size],
                parent_hash=parent_hash,
                extra_keys=extra_keys,
            )
            block_hashes.append(parent_hash)
        return len(self.block_pool.get_cached_blocks(block_hashes)) * self.block_size

    def get_layer_state(self, request_id: str) -> LayerKVState:
        return LayerKVState(
            layer_id=self.layer_id,
            blocks=self.request_blocks[request_id],
        )

    def release_request(self, request_id: str) -> list[int]:
        blocks = self.request_blocks.pop(request_id)
        self.num_cached_blocks.pop(request_id, None)
        block_ids: list[int] = []
        for block in reversed(blocks):
            block.alive = False
            block_ids.append(block.block_id)
        self.block_pool.release_many(block_ids)
        return block_ids

    def total_tokens(self, request_id: str) -> int:
        blocks = self.request_blocks.get(request_id, [])
        if not blocks:
            return 0
        return blocks[-1].token_end

    def _update_tail_token_end(self, *, request_id: str, total_tokens: int) -> None:
        blocks = self.request_blocks.get(request_id, [])
        if blocks:
            blocks[-1].token_end = total_tokens


def _required_block_count(total_tokens: int, block_size: int) -> int:
    if total_tokens == 0:
        return 0
    return (total_tokens + block_size - 1) // block_size


def _logical_block_from_physical(
    *,
    physical_block: KVCacheBlock,
    request_id: str,
    layer_id: int,
    token_start: int,
    token_end: int,
    block_size: int,
) -> KVBlock:
    return KVBlock(
        block_id=physical_block.block_id,
        seq_id=request_id,
        layer_id=layer_id,
        token_start=token_start,
        token_end=token_end,
        capacity=block_size,
        block_hash=physical_block.block_hash,
    )
