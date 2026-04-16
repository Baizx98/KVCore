"""Worker-side block table metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BlockTable:
    """Map KV-manager block ids to kernel-consumable block ids."""

    manager_block_size: int
    kernel_block_size: int
    manager_block_ids: list[int]

    def __post_init__(self) -> None:
        if self.manager_block_size <= 0:
            raise ValueError("manager_block_size must be positive")
        if self.kernel_block_size <= 0:
            raise ValueError("kernel_block_size must be positive")
        if self.manager_block_size % self.kernel_block_size != 0:
            raise ValueError("manager_block_size must be divisible by kernel_block_size")

    @property
    def kernel_blocks_per_manager_block(self) -> int:
        return self.manager_block_size // self.kernel_block_size

    def map_to_kernel_blocks(self) -> list[int]:
        kernel_block_ids: list[int] = []
        factor = self.kernel_blocks_per_manager_block
        for manager_block_id in self.manager_block_ids:
            start = manager_block_id * factor
            kernel_block_ids.extend(range(start, start + factor))
        return kernel_block_ids


@dataclass(slots=True)
class MultiGroupBlockTable:
    """Container for future multi-group support; currently one group per layer."""

    group_tables: dict[int, BlockTable]

    def map_to_kernel_blocks(self) -> dict[int, list[int]]:
        return {
            group_id: block_table.map_to_kernel_blocks()
            for group_id, block_table in self.group_tables.items()
        }
