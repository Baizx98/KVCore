from __future__ import annotations

import numpy as np
import torch

from kvcore.kv.block_table import (
    PAD_SLOT_ID,
    BlockTable,
    MultiGroupBlockTable,
)


def test_block_table_row_ops_and_commit() -> None:
    table = BlockTable(
        block_size=2,
        max_num_reqs=3,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cpu"),
    )

    table.add_row([3, 4], row_idx=0)
    table.append_row([5], row_idx=0)
    table.add_row([7], row_idx=1)
    table.move_row(0, 2)
    table.swap_row(1, 2)
    table.commit_block_table(num_reqs=3)

    assert table.num_blocks_per_row.tolist() == [3, 3, 1]
    assert table.get_numpy_array()[0, :3].tolist() == [3, 4, 5]
    assert table.get_numpy_array()[1, :3].tolist() == [3, 4, 5]
    assert table.get_numpy_array()[2, :1].tolist() == [7]
    assert table.get_device_tensor(num_reqs=3).shape == (3, 4)


def test_block_table_tracks_logical_block_indices() -> None:
    table = BlockTable(
        block_size=2,
        max_num_reqs=1,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cpu"),
    )

    table.add_row([3, 7], row_idx=0, block_indices=[0, 3])
    table.commit_block_table(num_reqs=1)

    assert table.get_device_tensor(num_reqs=1)[0, :2].tolist() == [3, 7]
    assert table.get_block_indices_device_tensor(num_reqs=1)[0, :2].tolist() == [0, 3]
    assert table.get_num_blocks_device_tensor(num_reqs=1).tolist() == [2]


def test_block_table_hybrid_kernel_block_mapping() -> None:
    mapped = BlockTable.map_to_kernel_blocks(
        np.array([0, 2], dtype=np.int32),
        blocks_per_kv_block=2,
        kernel_block_arange=np.arange(0, 2).reshape(1, -1),
    )

    assert mapped.tolist() == [0, 1, 4, 5]


def test_block_table_compute_slot_mapping() -> None:
    table = BlockTable(
        block_size=2,
        max_num_reqs=2,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cpu"),
    )
    table.add_row([3, 4], row_idx=0)
    table.add_row([5], row_idx=1)
    table.commit_block_table(num_reqs=2)

    table.compute_slot_mapping(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 4, 6], dtype=torch.int64),
        positions=torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int64),
    )

    assert table.slot_mapping.gpu[:6].tolist() == [6, 7, 8, 9, 10, 11]
    assert table.slot_mapping.gpu[6:].tolist() == [PAD_SLOT_ID, PAD_SLOT_ID]


def test_multi_group_block_table_matches_vllm_style_interface() -> None:
    tables = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=8,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[2, 4],
    )

    tables.add_row(([1, 2], [3]), row_idx=0)
    tables.append_row(([4], [5]), row_idx=0)
    tables.commit_block_table(num_reqs=1)

    assert tables[0].get_numpy_array()[0, :3].tolist() == [1, 2, 4]
    assert tables[1].get_numpy_array()[0, :2].tolist() == [3, 5]
