from kvcore.kv import BlockAllocator, RequestKVView


def test_block_allocator_reuses_released_ids() -> None:
    allocator = BlockAllocator()

    first = allocator.allocate_block_id()
    second = allocator.allocate_block_id()
    allocator.release_block_id(first)

    reused = allocator.allocate_block_id()

    assert second == 1
    assert reused == first


def test_request_kv_view_grows_and_releases() -> None:
    allocator = BlockAllocator()
    view = RequestKVView.from_token_count(
        seq_id="req-1",
        total_tokens=17,
        num_layers=2,
        block_size=16,
        allocator=allocator,
    )

    assert view.total_block_count == 4
    assert view.layer_states[0].blocks[-1].token_end == 17

    view.update_total_tokens(33, allocator)

    assert view.total_block_count == 6
    assert view.layer_states[0].blocks[-1].token_start == 32
    assert view.layer_states[0].blocks[-1].token_end == 33

    view.release(allocator)

    assert view.total_block_count == 0
    assert view.total_tokens == 0
