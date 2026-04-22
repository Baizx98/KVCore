from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NewType

from kvcore.utils.request import Request

BlockHash = NewType("BlockHash", bytes)
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)


def make_block_hash_with_group_id(
    block_hash: BlockHash,
    group_id: int,
) -> BlockHashWithGroupId:
    if group_id < 0:
        raise ValueError(f"group_id must be non-negative, got {group_id}")
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    return int.from_bytes(key[-4:], "big", signed=False)


def hash_block_tokens(
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    hasher = hashlib.sha256()
    if parent_block_hash is not None:
        hasher.update(parent_block_hash)
    hasher.update(tuple(curr_block_token_ids).__repr__().encode())
    if extra_keys is not None:
        hasher.update(extra_keys.__repr__().encode())
    return BlockHash(hasher.digest())


def get_request_block_hasher(block_size: int) -> Callable[[Request], list[BlockHash]]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    def request_block_hasher(request: Request) -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * block_size
        if start_token_idx + block_size > request.num_tokens:
            return []

        prev_block_hash = request.block_hashes[-1] if request.block_hashes else None
        new_block_hashes: list[BlockHash] = []
        while start_token_idx + block_size <= request.num_tokens:
            end_token_idx = start_token_idx + block_size
            block_hash = hash_block_tokens(
                prev_block_hash,
                request.all_token_ids[start_token_idx:end_token_idx],
                (request.cache_salt,) if start_token_idx == 0 and request.cache_salt else None,
            )
            new_block_hashes.append(block_hash)
            prev_block_hash = block_hash
            start_token_idx = end_token_idx
        return new_block_hashes

    return request_block_hasher


@dataclass(slots=True)
class KVBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: BlockHashWithGroupId | None = None
    prev_free_block: KVBlock | None = None
    next_free_block: KVBlock | None = None
    is_null: bool = False

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashWithGroupId) -> None:
        if self._block_hash is not None:
            raise ValueError(f"Block {self.block_id} already has a hash")
        self._block_hash = block_hash

    def reset_hash(self) -> None:
        self._block_hash = None

    def __repr__(self) -> str:
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            "KVBlock("
            f"block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"block_hash={self._block_hash!r}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id}, "
            f"is_null={self.is_null}"
            ")"
        )


class FreeKVBlockQueue:
    def __init__(self, blocks: list[KVBlock]) -> None:
        self.num_free_blocks = 0
        self.fake_free_list_head = KVBlock(block_id=-1)
        self.fake_free_list_tail = KVBlock(block_id=-1)

        self.fake_free_list_head.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = self.fake_free_list_head
        if blocks:
            self.append_n(blocks)

    def popleft(self) -> KVBlock:
        if self.num_free_blocks == 0:
            raise ValueError("No free blocks available")
        return self.popleft_n(1)[0]

    def popleft_n(self, n: int) -> list[KVBlock]:
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if n == 0:
            return []
        if n > self.num_free_blocks:
            raise ValueError(f"Cannot pop {n} blocks from {self.num_free_blocks} free blocks")

        current = self.fake_free_list_head.next_free_block
        ret: list[KVBlock] = []
        for _ in range(n):
            assert current is not None
            ret.append(current)
            next_block = current.next_free_block
            current.prev_free_block = None
            current.next_free_block = None
            current = next_block

        assert current is not None
        self.fake_free_list_head.next_free_block = current
        current.prev_free_block = self.fake_free_list_head
        self.num_free_blocks -= n
        return ret

    def remove(self, block: KVBlock) -> None:
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"remove() called on block not in free list: {block}")

        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVBlock) -> None:
        self.append_n([block])

    def append_n(self, blocks: list[KVBlock]) -> None:
        if not blocks:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None
        for block in blocks:
            if block.prev_free_block is not None or block.next_free_block is not None:
                raise RuntimeError(f"append() called on block already in a free list: {block}")
            last_block.next_free_block = block
            block.prev_free_block = last_block
            last_block = block

        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block
        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVBlock]:
        blocks: list[KVBlock] = []
        current = self.fake_free_list_head.next_free_block
        while current is not None and current is not self.fake_free_list_tail:
            blocks.append(current)
            current = current.next_free_block
        return blocks
