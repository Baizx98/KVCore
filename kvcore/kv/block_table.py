from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

NULL_BLOCK_ID = 0
PAD_SLOT_ID = -1
PAD_BLOCK_INDEX = -1


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


class CpuGpuBuffer:
    """Small local equivalent of vLLM's CPU/GPU staged tensor buffer."""

    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        self.cpu = torch.zeros(*size, dtype=dtype)
        if pin_memory and torch.cuda.is_available():
            self.cpu = self.cpu.pin_memory()
        self.gpu = torch.zeros(*size, dtype=dtype, device=device)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, num_rows: int | None = None) -> None:
        if num_rows is None or self.cpu.dim() == 1:
            self.gpu.copy_(self.cpu, non_blocking=self.cpu.is_pinned())
            return
        self.gpu[:num_rows].copy_(self.cpu[:num_rows], non_blocking=self.cpu.is_pinned())


class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int | None = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if max_num_reqs <= 0:
            raise ValueError(f"max_num_reqs must be positive, got {max_num_reqs}")
        if max_num_blocks_per_req <= 0:
            raise ValueError(
                "max_num_blocks_per_req must be positive, "
                f"got {max_num_blocks_per_req}"
            )
        if max_num_batched_tokens <= 0:
            raise ValueError(
                "max_num_batched_tokens must be positive, "
                f"got {max_num_batched_tokens}"
            )

        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        kernel_block_size = block_size if kernel_block_size is None else kernel_block_size
        if kernel_block_size == block_size:
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"block_size {block_size} evenly"
                )
            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block
        self.block_table = self._make_buffer(
            self.max_num_reqs,
            self.max_num_blocks_per_req,
            dtype=torch.int32,
        )
        self.block_indices = self._make_buffer(
            self.max_num_reqs,
            self.max_num_blocks_per_req,
            dtype=torch.int32,
        )
        self.block_indices.cpu.fill_(PAD_BLOCK_INDEX)
        self.block_indices.gpu.fill_(PAD_BLOCK_INDEX)
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_blocks_per_row_gpu = torch.zeros(
            (max_num_reqs,),
            dtype=torch.int32,
            device=device,
        )
        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens,
            dtype=torch.int64,
        )

        self._kernel_block_arange = (
            np.arange(0, self.blocks_per_kv_block).reshape(1, -1)
            if self.use_hybrid_blocks
            else None
        )

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
        block_indices: list[int] | None = None,
    ) -> None:
        if not block_ids:
            return
        self._check_row_idx(row_idx)
        if block_indices is None:
            start_block_index = int(self.num_blocks_per_row[row_idx])
            block_indices = list(
                range(start_block_index, start_block_index + len(block_ids))
            )
        if len(block_indices) != len(block_ids):
            raise ValueError(
                "block_indices must have the same length as block_ids, "
                f"got {len(block_indices)} and {len(block_ids)}"
            )
        mapped_block_ids: Sequence[int] | np.ndarray = block_ids
        mapped_block_indices: Sequence[int] | np.ndarray = block_indices
        if self.use_hybrid_blocks:
            assert self._kernel_block_arange is not None
            mapped_block_ids = self.map_to_kernel_blocks(
                np.array(block_ids, dtype=np.int32),
                self.blocks_per_kv_block,
                self._kernel_block_arange,
            )
            mapped_block_indices = np.repeat(
                np.array(block_indices, dtype=np.int32),
                self.blocks_per_kv_block,
            )

        num_blocks = len(mapped_block_ids)
        start = int(self.num_blocks_per_row[row_idx])
        end = start + num_blocks
        if end > self.max_num_blocks_per_req:
            raise ValueError(
                f"row {row_idx} cannot hold {end} blocks; "
                f"capacity is {self.max_num_blocks_per_req}"
            )
        self.block_table.np[row_idx, start:end] = mapped_block_ids
        self.block_indices.np[row_idx, start:end] = mapped_block_indices
        self.num_blocks_per_row[row_idx] = end

    def add_row(
        self,
        block_ids: list[int],
        row_idx: int,
        block_indices: list[int] | None = None,
    ) -> None:
        self.clear_row(row_idx)
        self.append_row(block_ids, row_idx, block_indices)

    def clear_row(self, row_idx: int) -> None:
        self._check_row_idx(row_idx)
        num_blocks = int(self.num_blocks_per_row[row_idx])
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = NULL_BLOCK_ID
            self.block_indices.np[row_idx, :num_blocks] = PAD_BLOCK_INDEX
        self.num_blocks_per_row[row_idx] = 0

    def move_row(self, src: int, tgt: int) -> None:
        self._check_row_idx(src)
        self._check_row_idx(tgt)
        num_blocks = int(self.num_blocks_per_row[src])
        self.block_table.np[tgt, :num_blocks] = self.block_table.np[src, :num_blocks]
        self.block_indices.np[tgt, :num_blocks] = self.block_indices.np[src, :num_blocks]
        if num_blocks < self.max_num_blocks_per_req:
            self.block_table.np[tgt, num_blocks:] = NULL_BLOCK_ID
            self.block_indices.np[tgt, num_blocks:] = PAD_BLOCK_INDEX
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        self._check_row_idx(src)
        self._check_row_idx(tgt)
        self.num_blocks_per_row[[src, tgt]] = self.num_blocks_per_row[[tgt, src]]
        self.block_table.np[[src, tgt]] = self.block_table.np[[tgt, src]]
        self.block_indices.np[[src, tgt]] = self.block_indices.np[[tgt, src]]

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if num_reqs < 0 or num_reqs > self.max_num_reqs:
            raise ValueError(f"num_reqs must be in [0, {self.max_num_reqs}], got {num_reqs}")

        query_start_loc_cpu = query_start_loc.detach().cpu().tolist()
        positions_cpu = positions.detach().cpu().tolist()
        num_tokens = len(positions_cpu)
        if num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                f"positions has {num_tokens} tokens but capacity is "
                f"{self.max_num_batched_tokens}"
        )

        self.slot_mapping.cpu.fill_(PAD_SLOT_ID)
        for req_idx in range(num_reqs):
            start = int(query_start_loc_cpu[req_idx])
            end = int(query_start_loc_cpu[req_idx + 1])
            for token_idx in range(start, end):
                position = int(positions_cpu[token_idx])
                block_index = position // self.block_size
                block_offset = position % self.block_size
                if block_index >= self.max_num_blocks_per_req:
                    self.slot_mapping.np[token_idx] = PAD_SLOT_ID
                    continue
                block_number = int(self.block_table.np[req_idx, block_index])
                slot_id = block_number * self.block_size + block_offset
                self.slot_mapping.np[token_idx] = slot_id
        self.slot_mapping.copy_to_gpu()

    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu(num_reqs)
        self.block_indices.copy_to_gpu(num_reqs)
        self.num_blocks_per_row_gpu[:num_reqs].copy_(
            torch.from_numpy(self.num_blocks_per_row[:num_reqs]),
            non_blocking=False,
        )

    def clear(self) -> None:
        self.block_table.gpu.fill_(NULL_BLOCK_ID)
        self.block_table.cpu.fill_(NULL_BLOCK_ID)
        self.block_indices.gpu.fill_(PAD_BLOCK_INDEX)
        self.block_indices.cpu.fill_(PAD_BLOCK_INDEX)
        self.num_blocks_per_row.fill(0)
        self.num_blocks_per_row_gpu.fill_(0)

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids
        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )
        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        return self.block_table.gpu[:num_reqs]

    def get_block_indices_device_tensor(self, num_reqs: int) -> torch.Tensor:
        return self.block_indices.gpu[:num_reqs]

    def get_num_blocks_device_tensor(self, num_reqs: int) -> torch.Tensor:
        return self.num_blocks_per_row_gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        return self.block_table.np

    def _make_buffer(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def _check_row_idx(self, row_idx: int) -> None:
        if row_idx < 0 or row_idx >= self.max_num_reqs:
            raise IndexError(f"row_idx must be in [0, {self.max_num_reqs}), got {row_idx}")


class MultiGroupBlockTable:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int] | None = None,
        max_num_blocks: list[int] | None = None,
    ) -> None:
        kernel_block_sizes = kernel_block_sizes or list(block_sizes)
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        if max_num_blocks is None:
            max_num_blocks = [cdiv(max_model_len, block_size) for block_size in block_sizes]
        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        self.block_tables = [
            BlockTable(
                block_size=block_size,
                max_num_reqs=max_num_reqs,
                max_num_blocks_per_req=max_num_blocks_per_req,
                max_num_batched_tokens=max_num_batched_tokens,
                pin_memory=pin_memory,
                device=device,
                kernel_block_size=kernel_block_size,
            )
            for block_size, kernel_block_size, max_num_blocks_per_req in zip(
                block_sizes,
                kernel_block_sizes,
                max_num_blocks,
                strict=True,
            )
        ]

    def append_row(
        self,
        block_ids: tuple[list[int], ...],
        row_idx: int,
        block_indices: tuple[list[int], ...] | None = None,
    ) -> None:
        for idx, block_table in enumerate(self.block_tables):
            layer_block_indices = None if block_indices is None else block_indices[idx]
            block_table.append_row(block_ids[idx], row_idx, layer_block_indices)

    def add_row(
        self,
        block_ids: tuple[list[int], ...],
        row_idx: int,
        block_indices: tuple[list[int], ...] | None = None,
    ) -> None:
        for idx, block_table in enumerate(self.block_tables):
            layer_block_indices = None if block_indices is None else block_indices[idx]
            block_table.add_row(block_ids[idx], row_idx, layer_block_indices)

    def clear_row(self, row_idx: int) -> None:
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> BlockTable:
        return self.block_tables[idx]

    def __len__(self) -> int:
        return len(self.block_tables)


__all__ = [
    "BlockTable",
    "CpuGpuBuffer",
    "MultiGroupBlockTable",
    "NULL_BLOCK_ID",
    "PAD_BLOCK_INDEX",
    "PAD_SLOT_ID",
]
