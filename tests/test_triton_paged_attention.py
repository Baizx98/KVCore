from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kvcore.kv.block_table import MultiGroupBlockTable
from kvcore.model.attn_backend.torch_paged import TorchPagedAttentionBackend
from kvcore.model.attn_backend.triton_paged import TritonPagedAttentionBackend
from kvcore.model.kv_runtime import PagedAttentionMetadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton paged attention requires CUDA",
)


def make_metadata(
    *,
    kv_cache_tensor: torch.Tensor,
    block_ids: list[int],
    flat_positions: list[int],
    context_len: int,
    query_len: int,
) -> PagedAttentionMetadata:
    device = kv_cache_tensor.device
    block_tables = MultiGroupBlockTable(
        max_num_reqs=1,
        max_model_len=16,
        max_num_batched_tokens=max(1, query_len),
        pin_memory=False,
        device=device,
        block_sizes=[2],
        kernel_block_sizes=[2],
    )
    block_tables.add_row((block_ids,), 0)
    block_tables.commit_block_table(1)

    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32, device=device)
    flat_positions_tensor = torch.tensor(flat_positions, dtype=torch.int32, device=device)
    block_tables[0].compute_slot_mapping(1, query_start_loc, flat_positions_tensor)

    return PagedAttentionMetadata(
        kv_cache_tensor=kv_cache_tensor,
        block_tables=block_tables,
        slot_mapping={0: block_tables[0].slot_mapping.gpu[:query_len]},
        query_start_loc=query_start_loc,
        seq_lens=torch.tensor([context_len + query_len], dtype=torch.int32, device=device),
        context_lens=torch.tensor([context_len], dtype=torch.int32, device=device),
        query_lens=torch.tensor([query_len], dtype=torch.int32, device=device),
        flat_positions=flat_positions_tensor,
        token_request_indices=torch.zeros(query_len, dtype=torch.int32, device=device),
        num_reqs=1,
        num_scheduled_tokens=query_len,
        num_prefill_reqs=1 if context_len == 0 else 0,
        num_decode_reqs=0 if context_len == 0 else 1,
        max_query_len=query_len,
        max_seq_len=context_len + query_len,
    )


def make_multi_request_metadata(
    *,
    kv_cache_tensor: torch.Tensor,
    block_rows: list[list[int]],
    flat_positions: list[int],
    query_lens: list[int],
    block_size: int = 2,
) -> PagedAttentionMetadata:
    device = kv_cache_tensor.device
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

    return PagedAttentionMetadata(
        kv_cache_tensor=kv_cache_tensor,
        block_tables=block_tables,
        slot_mapping={0: block_tables[0].slot_mapping.gpu[: len(flat_positions)]},
        query_start_loc=query_start_loc_tensor,
        seq_lens=torch.tensor(query_lens, dtype=torch.int32, device=device),
        context_lens=torch.zeros(len(query_lens), dtype=torch.int32, device=device),
        query_lens=torch.tensor(query_lens, dtype=torch.int32, device=device),
        flat_positions=flat_positions_tensor,
        token_request_indices=torch.tensor(token_request_indices, dtype=torch.int32, device=device),
        num_reqs=len(block_rows),
        num_scheduled_tokens=len(flat_positions),
        num_prefill_reqs=len(query_lens),
        num_decode_reqs=0,
        max_query_len=max(query_lens, default=0),
        max_seq_len=max(query_lens, default=0),
    )


def test_triton_paged_attention_matches_prefill_reference() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = TritonPagedAttentionBackend()
    kv_cache_tensor = torch.zeros((2, 8, 2, 1, 16), dtype=torch.float16, device=device)

    query = torch.randn((1, 1, 4, 16), dtype=torch.float16, device=device)
    key = torch.randn((1, 1, 4, 16), dtype=torch.float16, device=device)
    value = torch.randn((1, 1, 4, 16), dtype=torch.float16, device=device)
    metadata = make_metadata(
        kv_cache_tensor=kv_cache_tensor,
        block_ids=[1, 2],
        flat_positions=[0, 1, 2, 3],
        context_len=0,
        query_len=4,
    )

    output = backend.forward(
        query,
        key,
        value,
        num_kv_heads=1,
        scaling=16**-0.5,
        is_causal=True,
        attn_metadata=metadata,
        layer_idx=0,
    )
    reference = F.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        scale=16**-0.5,
    )

    assert torch.allclose(output, reference, atol=1e-2, rtol=1e-2)


def test_triton_paged_attention_matches_torch_paged_multi_request_gqa() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    triton_backend = TritonPagedAttentionBackend()
    torch_backend = TorchPagedAttentionBackend()

    query = torch.randn((1, 4, 5, 16), dtype=torch.float16, device=device)
    key = torch.randn((1, 2, 5, 16), dtype=torch.float16, device=device)
    value = torch.randn((1, 2, 5, 16), dtype=torch.float16, device=device)
    triton_metadata = make_multi_request_metadata(
        kv_cache_tensor=torch.zeros((2, 16, 2, 2, 16), dtype=torch.float16, device=device),
        block_rows=[[1, 2], [3]],
        flat_positions=[0, 1, 2, 0, 1],
        query_lens=[3, 2],
    )
    torch_metadata = make_multi_request_metadata(
        kv_cache_tensor=torch.zeros((2, 16, 2, 2, 16), dtype=torch.float16, device=device),
        block_rows=[[1, 2], [3]],
        flat_positions=[0, 1, 2, 0, 1],
        query_lens=[3, 2],
    )

    triton_output = triton_backend.forward(
        query,
        key,
        value,
        num_kv_heads=2,
        scaling=16**-0.5,
        is_causal=True,
        attn_metadata=triton_metadata,
        layer_idx=0,
    )
    torch_output = torch_backend.forward(
        query,
        key,
        value,
        num_kv_heads=2,
        scaling=16**-0.5,
        is_causal=True,
        attn_metadata=torch_metadata,
        layer_idx=0,
    )

    assert torch.allclose(triton_output, torch_output, atol=2e-2, rtol=2e-2)


def test_triton_paged_attention_reads_previous_decode_cache() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = TritonPagedAttentionBackend()
    kv_cache_tensor = torch.zeros((2, 8, 2, 1, 16), dtype=torch.float16, device=device)

    prefill_query = torch.randn((1, 1, 4, 16), dtype=torch.float16, device=device)
    prefill_key = torch.randn((1, 1, 4, 16), dtype=torch.float16, device=device)
    prefill_value = torch.randn((1, 1, 4, 16), dtype=torch.float16, device=device)
    prefill_metadata = make_metadata(
        kv_cache_tensor=kv_cache_tensor,
        block_ids=[1, 2],
        flat_positions=[0, 1, 2, 3],
        context_len=0,
        query_len=4,
    )
    backend.forward(
        prefill_query,
        prefill_key,
        prefill_value,
        num_kv_heads=1,
        scaling=16**-0.5,
        is_causal=True,
        attn_metadata=prefill_metadata,
        layer_idx=0,
    )

    decode_query = torch.randn((1, 1, 1, 16), dtype=torch.float16, device=device)
    decode_key = torch.randn((1, 1, 1, 16), dtype=torch.float16, device=device)
    decode_value = torch.randn((1, 1, 1, 16), dtype=torch.float16, device=device)
    decode_metadata = make_metadata(
        kv_cache_tensor=kv_cache_tensor,
        block_ids=[1, 2, 3],
        flat_positions=[4],
        context_len=4,
        query_len=1,
    )
    output = backend.forward(
        decode_query,
        decode_key,
        decode_value,
        num_kv_heads=1,
        scaling=16**-0.5,
        is_causal=True,
        attn_metadata=decode_metadata,
        layer_idx=0,
    )

    all_key = torch.cat([prefill_key, decode_key], dim=2)
    all_value = torch.cat([prefill_value, decode_value], dim=2)
    scores = torch.matmul(
        decode_query.to(torch.float32),
        all_key.transpose(-1, -2).to(torch.float32),
    ) * (16**-0.5)
    probs = torch.softmax(scores, dim=-1)
    reference = torch.matmul(probs, all_value.to(torch.float32)).to(torch.float16)

    assert torch.allclose(output, reference, atol=2e-2, rtol=2e-2)
