from __future__ import annotations

import torch

from kvcore.model.kv_runtime import PagedAttentionMetadata

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - runtime guarded
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _paged_write_kernel(
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        slot_mapping_ptr,
        num_tokens,
        num_kv_heads,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_cb,
        stride_cs,
        stride_ch,
        stride_cd,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        kv_head_idx = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_D)

        if token_idx >= num_tokens or kv_head_idx >= num_kv_heads:
            return

        slot = tl.load(slot_mapping_ptr + token_idx)
        if slot < 0:
            return

        block_id = slot // BLOCK_SIZE
        block_offset = slot % BLOCK_SIZE
        dim_mask = offs_d < HEAD_DIM

        key_src_ptrs = (
            key_ptr
            + token_idx * stride_kt
            + kv_head_idx * stride_kh
            + offs_d * stride_kd
        )
        value_src_ptrs = (
            value_ptr
            + token_idx * stride_vt
            + kv_head_idx * stride_vh
            + offs_d * stride_vd
        )
        key_dst_ptrs = (
            key_cache_ptr
            + block_id * stride_cb
            + block_offset * stride_cs
            + kv_head_idx * stride_ch
            + offs_d * stride_cd
        )
        value_dst_ptrs = (
            value_cache_ptr
            + block_id * stride_cb
            + block_offset * stride_cs
            + kv_head_idx * stride_ch
            + offs_d * stride_cd
        )

        tl.store(key_dst_ptrs, tl.load(key_src_ptrs, mask=dim_mask, other=0.0), mask=dim_mask)
        tl.store(
            value_dst_ptrs,
            tl.load(value_src_ptrs, mask=dim_mask, other=0.0),
            mask=dim_mask,
        )


    @triton.jit
    def _paged_attention_kernel(
        query_ptr,
        output_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_table_ptr,
        token_request_indices_ptr,
        flat_positions_ptr,
        scale,
        num_tokens,
        num_query_heads,
        num_queries_per_kv,
        block_table_stride,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_ot,
        stride_oh,
        stride_od,
        stride_cb,
        stride_cs,
        stride_ch,
        stride_cd,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        query_head_idx = tl.program_id(1)
        if token_idx >= num_tokens or query_head_idx >= num_query_heads:
            return

        request_idx = tl.load(token_request_indices_ptr + token_idx)
        query_position = tl.load(flat_positions_ptr + token_idx)
        kv_head_idx = query_head_idx // num_queries_per_kv

        offs_d = tl.arange(0, BLOCK_D)
        dim_mask = offs_d < HEAD_DIM
        query_ptrs = (
            query_ptr
            + token_idx * stride_qt
            + query_head_idx * stride_qh
            + offs_d * stride_qd
        )
        query = tl.load(query_ptrs, mask=dim_mask, other=0.0).to(tl.float32)

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        start = 0
        while start <= query_position:
            seq_positions = start + tl.arange(0, BLOCK_N)
            seq_mask = seq_positions <= query_position

            logical_block_indices = seq_positions // BLOCK_SIZE
            block_ids = tl.load(
                block_table_ptr + request_idx * block_table_stride + logical_block_indices,
                mask=seq_mask,
                other=0,
            )
            block_offsets = seq_positions % BLOCK_SIZE
            cache_ptrs = (
                block_ids[:, None] * stride_cb
                + block_offsets[:, None] * stride_cs
                + kv_head_idx * stride_ch
                + offs_d[None, :] * stride_cd
            )
            key = tl.load(
                key_cache_ptr + cache_ptrs,
                mask=seq_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                value_cache_ptr + cache_ptrs,
                mask=seq_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            scores = tl.sum(key * query[None, :], axis=1) * scale
            scores = tl.where(seq_mask, scores, float("-inf"))

            block_max = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, block_max)
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)
            acc = acc * alpha + tl.sum(value * p[:, None], axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new
            start += BLOCK_N

        output = acc / tl.maximum(l_i, 1e-20)
        output_ptrs = (
            output_ptr
            + token_idx * stride_ot
            + query_head_idx * stride_oh
            + offs_d * stride_od
        )
        tl.store(output_ptrs, output, mask=dim_mask)


class TritonPagedAttentionBackend:
    def __init__(self) -> None:
        if triton is None or tl is None:
            raise ImportError("triton is required for TritonPagedAttentionBackend")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        num_kv_heads: int,
        scaling: float,
        is_causal: bool,
        attn_metadata: object | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        if not is_causal:
            raise NotImplementedError(
                "Triton paged attention only supports decoder causal attention"
            )
        if layer_idx is None:
            raise ValueError("layer_idx is required for Triton paged attention")
        metadata = self._require_metadata(attn_metadata)
        if query.device.type != "cuda":
            raise RuntimeError("Triton paged attention requires CUDA tensors")
        if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
            raise ValueError(
                "Triton paged attention expects flattened runtime inputs with batch size 1, "
                f"got query shape {tuple(query.shape)}"
            )

        query_states = query[0].transpose(0, 1).contiguous()
        key_states = key[0].transpose(0, 1).contiguous()
        value_states = value[0].transpose(0, 1).contiguous()

        num_tokens, num_query_heads, head_dim = query_states.shape
        if num_tokens != metadata.num_scheduled_tokens:
            raise ValueError(
                "metadata.num_scheduled_tokens must match query tokens, "
                f"got {metadata.num_scheduled_tokens} and {num_tokens}"
            )
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_kv_heads, "
                f"got {num_query_heads} and {num_kv_heads}"
            )

        key_cache = metadata.kv_cache_tensor[0]
        value_cache = metadata.kv_cache_tensor[1]
        slot_mapping = metadata.slot_mapping[layer_idx][:num_tokens]
        block_table = metadata.block_tables[layer_idx].get_device_tensor(metadata.num_reqs)
        output = torch.empty_like(query_states)

        block_size = key_cache.shape[1]
        block_d = triton.next_power_of_2(head_dim)

        self._paged_write(
            key_states,
            value_states,
            key_cache,
            value_cache,
            slot_mapping,
            block_size,
            head_dim,
            block_d,
        )
        self._paged_attention(
            query_states,
            output,
            key_cache,
            value_cache,
            block_table,
            metadata.token_request_indices,
            metadata.flat_positions,
            scaling,
            block_size,
            head_dim,
            block_d,
            num_query_heads // num_kv_heads,
        )

        return output.transpose(0, 1).unsqueeze(0)

    @staticmethod
    def _require_metadata(attn_metadata: object | None) -> PagedAttentionMetadata:
        if not isinstance(attn_metadata, PagedAttentionMetadata):
            raise TypeError(
                "Triton paged attention requires PagedAttentionMetadata, "
                f"got {type(attn_metadata)!r}"
            )
        return attn_metadata

    @staticmethod
    def _paged_write(
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
        head_dim: int,
        block_d: int,
    ) -> None:
        grid = (key_states.shape[0], key_states.shape[1])
        _paged_write_kernel[grid](
            key_states,
            value_states,
            key_cache,
            value_cache,
            slot_mapping,
            key_states.shape[0],
            key_states.shape[1],
            key_states.stride(0),
            key_states.stride(1),
            key_states.stride(2),
            value_states.stride(0),
            value_states.stride(1),
            value_states.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
        )

    @staticmethod
    def _paged_attention(
        query_states: torch.Tensor,
        output: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        token_request_indices: torch.Tensor,
        flat_positions: torch.Tensor,
        scaling: float,
        block_size: int,
        head_dim: int,
        block_d: int,
        num_queries_per_kv: int,
    ) -> None:
        grid = (query_states.shape[0], query_states.shape[1])
        block_n = 16 if flat_positions.numel() <= 512 else 32
        _paged_attention_kernel[grid](
            query_states,
            output,
            key_cache,
            value_cache,
            block_table,
            token_request_indices,
            flat_positions,
            scaling,
            query_states.shape[0],
            query_states.shape[1],
            num_queries_per_kv,
            block_table.stride(0),
            query_states.stride(0),
            query_states.stride(1),
            query_states.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
        )


__all__ = [
    "TritonPagedAttentionBackend",
]
