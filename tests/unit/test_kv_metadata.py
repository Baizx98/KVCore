from kvcore.kv import BlockPool, BlockTable, KVManager


def test_block_pool_reuses_released_ids() -> None:
    pool = BlockPool(capacity=4)

    first = pool.allocate_block().block_id
    second = pool.allocate_block().block_id
    pool.release_block_id(first)
    third = pool.allocate_block().block_id
    reused = pool.allocate_block().block_id

    assert second == 2
    assert third == 3
    assert reused == first


def test_kv_manager_uses_one_single_type_manager_per_layer() -> None:
    manager = KVManager.from_model_config(num_layers=3, block_size=4, device="cpu")

    view = manager.register_request(request_id="req-1", total_tokens=5)

    assert set(manager.layer_managers) == {0, 1, 2}
    assert view.total_block_count == 6
    assert all(
        state.blocks[0].layer_id == layer_id for layer_id, state in view.layer_states.items()
    )

    manager.advance_request(request_id="req-1", total_tokens=9)

    assert view.total_block_count == 9
    assert view.layer_states[0].blocks[-1].token_start == 8
    assert view.layer_states[0].blocks[-1].token_end == 9

    manager.release_request("req-1")

    assert view.total_tokens == 0
    assert view.total_block_count == 0


def test_kv_manager_finds_common_prefix_hit_across_layer_managers() -> None:
    manager = KVManager.from_model_config(num_layers=2, block_size=2, device="cpu")
    manager.register_request(request_id="req-1", total_tokens=4)
    token_ids = [10, 11, 12, 13]

    manager.cache_request_blocks(request_id="req-1", token_ids=token_ids)

    assert manager.find_longest_cache_hit(token_ids=token_ids, max_length=4) == 4


def test_block_pool_tracks_cached_hashes_without_deduping() -> None:
    pool = BlockPool(capacity=4)
    first = pool.allocate_block()
    second = pool.allocate_block()

    pool.cache_full_blocks(blocks=[first, second], block_hashes=["same", "same"])

    assert sorted(pool.cached_block_hash_to_block["same"]) == [first.block_id, second.block_id]


def test_block_table_maps_manager_blocks_to_kernel_blocks() -> None:
    block_table = BlockTable(
        manager_block_size=32,
        kernel_block_size=16,
        manager_block_ids=[2, 5],
    )

    assert block_table.map_to_kernel_blocks() == [4, 5, 10, 11]
