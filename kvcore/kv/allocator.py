"""Logical block allocator for request-local KV metadata."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class BlockAllocator:
    """Allocate and recycle logical block identifiers."""

    _next_block_id: int = 0
    _free_block_ids: deque[int] = field(default_factory=deque)

    def allocate_block_id(self) -> int:
        if self._free_block_ids:
            return self._free_block_ids.popleft()
        block_id = self._next_block_id
        self._next_block_id += 1
        return block_id

    def release_block_id(self, block_id: int) -> None:
        self._free_block_ids.append(block_id)

    def release_many(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            self.release_block_id(block_id)
