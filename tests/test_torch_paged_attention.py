from __future__ import annotations

import torch

from kvcore.kv.block_table import MultiGroupBlockTable
from kvcore.model.attn_backend.torch_paged import TorchPagedAttentionBackend
from kvcore.model.kv_runtime import PagedAttentionMetadata


def make_metadata(
    *,
    kv_cache_tensor: torch.Tensor,
    block_rows: list[list[int]],
    flat_positions: list[int],
    query_lens: list[int],
    context_lens: list[int] | None = None,
    block_size: int = 2,
) -> PagedAttentionMetadata:
    device = kv_cache_tensor.device
    context_lens = context_lens or [0 for _ in query_lens]
    block_tables = MultiGroupBlockTable(
        max_num_reqs=len(block_rows),
        max_model_len=32,
        max_num_batched_tokens=max(1, len(flat_positions)),
        pin_memory=False,
        device=device,
        block_sizes=[block_size],
        kernel_block_sizes=[block_size],
    )
    for row_idx, block_ids in enumerate(block_rows):
        block_tables.add_row((block_ids,), row_idx)
    block_tables.commit_block_table(len(block_rows))

    query_start_loc = [0]
    token_request_indices: list[int] = []
    for request_idx, query_len in enumerate(query_lens):
        query_start_loc.append(query_start_loc[-1] + query_len)
        token_request_indices.extend([request_idx] * query_len)
    query_start_loc_tensor = torch.tensor(query_start_loc, dtype=torch.int32, device=device)
    flat_positions_tensor = torch.tensor(flat_positions, dtype=torch.int32, device=device)
    block_tables[0].compute_slot_mapping(
        len(block_rows),
        query_start_loc_tensor,
        flat_positions_tensor,
    )

    seq_lens = [
        context_len + query_len
        for context_len, query_len in zip(context_lens, query_lens, strict=True)
    ]
    return PagedAttentionMetadata(
        kv_cache_tensor=kv_cache_tensor,
        block_tables=block_tables,
        slot_mapping={0: block_tables[0].slot_mapping.gpu[: len(flat_positions)]},
        query_start_loc=query_start_loc_tensor,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        context_lens=torch.tensor(context_lens, dtype=torch.int32, device=device),
        query_lens=torch.tensor(query_lens, dtype=torch.int32, device=device),
        flat_positions=flat_positions_tensor,
        token_request_indices=torch.tensor(token_request_indices, dtype=torch.int32, device=device),
        num_reqs=len(block_rows),
        num_scheduled_tokens=len(flat_positions),
        num_prefill_reqs=sum(1 for context_len in context_lens if context_len == 0),
        num_decode_reqs=sum(1 for context_len in context_lens if context_len > 0),
        max_query_len=max(query_lens, default=0),
        max_seq_len=max(seq_lens, default=0),
    )


def make_sparse_decode_metadata(
    *,
    kv_cache_tensor: torch.Tensor,
    full_block_ids: list[int],
    read_block_ids: list[int],
    read_block_indices: list[int],
    position: int,
    block_size: int = 2,
) -> PagedAttentionMetadata:
    device = kv_cache_tensor.device
    read_tables = MultiGroupBlockTable(
        max_num_reqs=1,
        max_model_len=32,
        max_num_batched_tokens=1,
        pin_memory=False,
        device=device,
        block_sizes=[block_size],
        kernel_block_sizes=[block_size],
    )
    full_tables = MultiGroupBlockTable(
        max_num_reqs=1,
        max_model_len=32,
        max_num_batched_tokens=1,
        pin_memory=False,
        device=device,
        block_sizes=[block_size],
        kernel_block_sizes=[block_size],
    )
    read_tables.add_row((read_block_ids,), 0, (read_block_indices,))
    read_tables.commit_block_table(1)
    full_tables.add_row((full_block_ids,), 0)

    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    flat_positions_tensor = torch.tensor([position], dtype=torch.int32, device=device)
    full_tables[0].compute_slot_mapping(1, query_start_loc, flat_positions_tensor)

    return PagedAttentionMetadata(
        kv_cache_tensor=kv_cache_tensor,
        block_tables=read_tables,
        slot_mapping={0: full_tables[0].slot_mapping.gpu[:1]},
        query_start_loc=query_start_loc,
        seq_lens=torch.tensor([position + 1], dtype=torch.int32, device=device),
        context_lens=torch.tensor([position], dtype=torch.int32, device=device),
        query_lens=torch.tensor([1], dtype=torch.int32, device=device),
        flat_positions=flat_positions_tensor,
        token_request_indices=torch.zeros(1, dtype=torch.int32, device=device),
        num_reqs=1,
        num_scheduled_tokens=1,
        num_prefill_reqs=0,
        num_decode_reqs=1,
        max_query_len=1,
        max_seq_len=position + 1,
    )


def dense_flattened_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    flat_positions: list[int],
    query_lens: list[int],
    num_kv_heads: int,
    scaling: float,
) -> torch.Tensor:
    query_states = query[0].transpose(0, 1)
    key_states = key[0].transpose(0, 1)
    value_states = value[0].transpose(0, 1)
    num_query_heads = query_states.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    output = torch.empty_like(query_states)

    start = 0
    for query_len in query_lens:
        end = start + query_len
        req_keys = key_states[start:end]
        req_values = value_states[start:end]
        req_positions = flat_positions[start:end]
        for local_token_idx, position in enumerate(req_positions):
            token_idx = start + local_token_idx
            visible = [idx for idx, req_pos in enumerate(req_positions) if req_pos <= position]
            visible_tensor = torch.tensor(visible, dtype=torch.long, device=query.device)
            for query_head_idx in range(num_query_heads):
                kv_head_idx = query_head_idx // num_queries_per_kv
                keys = req_keys.index_select(0, visible_tensor)[:, kv_head_idx]
                values = req_values.index_select(0, visible_tensor)[:, kv_head_idx]
                scores = (
                    torch.matmul(
                        keys.to(torch.float32),
                        query_states[token_idx, query_head_idx].to(torch.float32),
                    )
                    * scaling
                )
                probs = torch.softmax(scores, dim=0)
                output[token_idx, query_head_idx] = torch.matmul(
                    probs,
                    values.to(torch.float32),
                ).to(output.dtype)
        start = end
    return output.transpose(0, 1).unsqueeze(0)


def test_torch_paged_attention_matches_multi_request_gqa_dense_reference() -> None:
    torch.manual_seed(0)
    backend = TorchPagedAttentionBackend()
    kv_cache_tensor = torch.zeros((2, 16, 2, 2, 8), dtype=torch.float32)
    query = torch.randn((1, 4, 5, 8))
    key = torch.randn((1, 2, 5, 8))
    value = torch.randn((1, 2, 5, 8))
    flat_positions = [0, 1, 2, 0, 1]
    query_lens = [3, 2]
    metadata = make_metadata(
        kv_cache_tensor=kv_cache_tensor,
        block_rows=[[1, 2], [3]],
        flat_positions=flat_positions,
        query_lens=query_lens,
    )

    output = backend.forward(
        query,
        key,
        value,
        num_kv_heads=2,
        scaling=8**-0.5,
        is_causal=True,
        attn_metadata=metadata,
        layer_idx=0,
    )
    reference = dense_flattened_reference(
        query,
        key,
        value,
        flat_positions=flat_positions,
        query_lens=query_lens,
        num_kv_heads=2,
        scaling=8**-0.5,
    )

    assert torch.allclose(output, reference, atol=1e-5, rtol=1e-5)


def test_torch_paged_attention_reads_previous_decode_cache() -> None:
    torch.manual_seed(0)
    backend = TorchPagedAttentionBackend()
    kv_cache_tensor = torch.zeros((2, 16, 2, 1, 8), dtype=torch.float32)

    prefill_key = torch.randn((1, 1, 4, 8))
    prefill_value = torch.randn((1, 1, 4, 8))
    prefill_metadata = make_metadata(
        kv_cache_tensor=kv_cache_tensor,
        block_rows=[[1, 2]],
        flat_positions=[0, 1, 2, 3],
        query_lens=[4],
    )
    backend.forward(
        torch.randn((1, 1, 4, 8)),
        prefill_key,
        prefill_value,
        num_kv_heads=1,
        scaling=8**-0.5,
        is_causal=True,
        attn_metadata=prefill_metadata,
        layer_idx=0,
    )

    decode_query = torch.randn((1, 1, 1, 8))
    decode_key = torch.randn((1, 1, 1, 8))
    decode_value = torch.randn((1, 1, 1, 8))
    decode_metadata = make_metadata(
        kv_cache_tensor=kv_cache_tensor,
        block_rows=[[1, 2, 3]],
        flat_positions=[4],
        query_lens=[1],
        context_lens=[4],
    )
    output = backend.forward(
        decode_query,
        decode_key,
        decode_value,
        num_kv_heads=1,
        scaling=8**-0.5,
        is_causal=True,
        attn_metadata=decode_metadata,
        layer_idx=0,
    )

    all_key = torch.cat([prefill_key, decode_key], dim=2)
    all_value = torch.cat([prefill_value, decode_value], dim=2)
    scores = torch.matmul(
        decode_query.to(torch.float32),
        all_key.transpose(-1, -2).to(torch.float32),
    ) * (8**-0.5)
    probs = torch.softmax(scores, dim=-1)
    reference = torch.matmul(probs, all_value.to(torch.float32))

    assert torch.allclose(output, reference, atol=1e-5, rtol=1e-5)


def test_torch_paged_attention_sparse_read_skips_hidden_blocks() -> None:
    torch.manual_seed(0)
    backend = TorchPagedAttentionBackend()
    kv_cache_tensor = torch.zeros((2, 16, 2, 1, 8), dtype=torch.float32)
    key_cache = kv_cache_tensor[0]
    value_cache = kv_cache_tensor[1]
    key_cache[1, 0:2, 0] = torch.randn((2, 8))
    key_cache[2, 0:2, 0] = torch.randn((2, 8))
    value_cache[1, 0:2, 0] = torch.randn((2, 8))
    value_cache[2, 0:2, 0] = torch.randn((2, 8))

    decode_query = torch.randn((1, 1, 1, 8))
    decode_key = torch.randn((1, 1, 1, 8))
    decode_value = torch.randn((1, 1, 1, 8))
    metadata = make_sparse_decode_metadata(
        kv_cache_tensor=kv_cache_tensor,
        full_block_ids=[1, 2, 3],
        read_block_ids=[1, 3],
        read_block_indices=[0, 2],
        position=4,
    )

    output = backend.forward(
        decode_query,
        decode_key,
        decode_value,
        num_kv_heads=1,
        scaling=8**-0.5,
        is_causal=True,
        attn_metadata=metadata,
        layer_idx=0,
    )

    visible_keys = torch.cat([key_cache[1, 0:2, 0], decode_key[0, 0]], dim=0)
    visible_values = torch.cat([value_cache[1, 0:2, 0], decode_value[0, 0]], dim=0)
    scores = torch.matmul(
        visible_keys.to(torch.float32),
        decode_query[0, 0, 0].to(torch.float32),
    ) * (8**-0.5)
    probs = torch.softmax(scores, dim=0)
    reference = torch.matmul(probs, visible_values.to(torch.float32)).view(1, 1, 1, 8)

    assert torch.allclose(output, reference, atol=1e-5, rtol=1e-5)
