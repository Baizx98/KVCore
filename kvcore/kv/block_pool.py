"""Physical KV block pool inspired by vLLM-style pre-created block management."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class PhysicalBlock:
    """A reusable physical KV block entry."""

    block_id: int
    ref_cnt: int = 0
    state: str = "free"


@dataclass(slots=True)
class BlockPool:
    """Manage reusable physical blocks through a free queue."""

    capacity: int = 4096
    blocks: dict[int, PhysicalBlock] = field(init=False)
    _free_block_ids: deque[int] = field(init=False)
    _next_block_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.blocks = {block_id: PhysicalBlock(block_id=block_id) for block_id in range(self.capacity)}
        self._free_block_ids = deque(range(self.capacity))
        self._next_block_id = self.capacity

    def allocate_block(self) -> PhysicalBlock:
        if self._free_block_ids:
            block_id = self._free_block_ids.popleft()
            block = self.blocks[block_id]
        else:
            block_id = self._next_block_id
            self._next_block_id += 1
            block = PhysicalBlock(block_id=block_id)
            self.blocks[block_id] = block
        block.ref_cnt = 1
        block.state = "used"
        return block

    def allocate_block_id(self) -> int:
        return self.allocate_block().block_id

    def free_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_cnt = 0
        block.state = "free"
        self._free_block_ids.append(block_id)

    def release_block_id(self, block_id: int) -> None:
        self.free_block(block_id)

    def release_many(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            self.free_block(block_id)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_ids)
