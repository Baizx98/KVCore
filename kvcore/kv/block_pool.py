"""Physical KV block pool with vLLM-style metadata hooks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import blake2b
from typing import TypeAlias

BlockHash: TypeAlias = str


@dataclass(slots=True)
class KVCacheBlock:
    """Reusable physical KV block metadata."""

    block_id: int
    ref_cnt: int = 0
    block_hash: BlockHash | None = None
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None
    is_null: bool = False
    state: str = "free"


PhysicalBlock = KVCacheBlock


@dataclass(slots=True)
class FreeKVCacheBlockQueue:
    """Doubly-linked free queue used for low-overhead block reuse."""

    head: KVCacheBlock | None = None
    tail: KVCacheBlock | None = None
    _block_ids: set[int] = field(default_factory=set)

    def popleft(self) -> KVCacheBlock:
        if self.head is None:
            raise IndexError("no free KV cache blocks available")
        block = self.head
        self.remove(block)
        return block

    def append(self, block: KVCacheBlock) -> None:
        if block.is_null or block.block_id in self._block_ids:
            return
        block.prev_free_block = self.tail
        block.next_free_block = None
        if self.tail is not None:
            self.tail.next_free_block = block
        else:
            self.head = block
        self.tail = block
        self._block_ids.add(block.block_id)

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            self.append(block)

    def remove(self, block: KVCacheBlock) -> None:
        if block.block_id not in self._block_ids:
            return
        prev_block = block.prev_free_block
        next_block = block.next_free_block
        if prev_block is not None:
            prev_block.next_free_block = next_block
        else:
            self.head = next_block
        if next_block is not None:
            next_block.prev_free_block = prev_block
        else:
            self.tail = prev_block
        block.prev_free_block = None
        block.next_free_block = None
        self._block_ids.remove(block.block_id)

    def __len__(self) -> int:
        return len(self._block_ids)


@dataclass(slots=True)
class BlockPool:
    """Own KV cache blocks, free ordering, and prefix-cache hash metadata."""

    capacity: int = 4096
    blocks: list[KVCacheBlock] = field(init=False)
    null_block: KVCacheBlock = field(init=False)
    free_block_queue: FreeKVCacheBlockQueue = field(init=False)
    cached_block_hash_to_block: dict[BlockHash, dict[int, KVCacheBlock]] = field(init=False)
    _next_block_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must reserve at least one null block")
        self.blocks = [KVCacheBlock(block_id=block_id) for block_id in range(self.capacity)]
        self.null_block = self.blocks[0]
        self.null_block.is_null = True
        self.null_block.state = "null"
        self.free_block_queue = FreeKVCacheBlockQueue()
        self.free_block_queue.append_n(self.blocks[1:])
        self.cached_block_hash_to_block = defaultdict(dict)
        self._next_block_id = self.capacity

    def allocate_block(self) -> KVCacheBlock:
        if len(self.free_block_queue) > 0:
            block = self.free_block_queue.popleft()
        else:
            block = KVCacheBlock(block_id=self._next_block_id)
            self._next_block_id += 1
            self.blocks.append(block)
        self._drop_cached_mapping(block)
        block.ref_cnt = 1
        block.state = "used"
        return block

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        return [self.allocate_block() for _ in range(num_blocks)]

    def allocate_block_id(self) -> int:
        return self.allocate_block().block_id

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            if block.is_null:
                continue
            self.free_block_queue.remove(block)
            block.ref_cnt += 1
            block.state = "used"

    def cache_full_blocks(
        self,
        *,
        blocks: list[KVCacheBlock],
        block_hashes: list[BlockHash],
    ) -> None:
        if len(blocks) != len(block_hashes):
            raise ValueError("blocks and block_hashes must have identical length")
        for block, block_hash in zip(blocks, block_hashes, strict=True):
            if block.is_null:
                continue
            block.block_hash = block_hash
            self.cached_block_hash_to_block[block_hash][block.block_id] = block

    def get_cached_blocks(self, block_hashes: list[BlockHash]) -> list[KVCacheBlock]:
        cached_blocks: list[KVCacheBlock] = []
        for block_hash in block_hashes:
            candidates = self.cached_block_hash_to_block.get(block_hash)
            if not candidates:
                break
            cached_blocks.append(next(iter(candidates.values())))
        return cached_blocks

    def free_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.is_null:
            return
        block.ref_cnt = 0
        block.state = "free"
        self.free_block_queue.append(block)

    def release_block_id(self, block_id: int) -> None:
        self.free_block(block_id)

    def release_many(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            self.free_block(block_id)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_queue)

    def _drop_cached_mapping(self, block: KVCacheBlock) -> None:
        if block.block_hash is None:
            return
        cached_blocks = self.cached_block_hash_to_block.get(block.block_hash)
        if cached_blocks is not None:
            cached_blocks.pop(block.block_id, None)
            if not cached_blocks:
                self.cached_block_hash_to_block.pop(block.block_hash, None)
        block.block_hash = None


def hash_block_tokens(
    *,
    token_ids: list[int],
    parent_hash: BlockHash | None = None,
    extra_keys: tuple[object, ...] = (),
) -> BlockHash:
    """Build a chained block hash for prefix-cache metadata."""

    digest = blake2b(digest_size=16)
    if parent_hash is not None:
        digest.update(parent_hash.encode("utf-8"))
    digest.update(repr(tuple(token_ids)).encode("utf-8"))
    if extra_keys:
        digest.update(repr(extra_keys).encode("utf-8"))
    return digest.hexdigest()
