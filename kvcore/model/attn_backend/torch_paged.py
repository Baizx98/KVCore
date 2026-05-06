from __future__ import annotations

import torch

from kvcore.model.kv_runtime import PagedAttentionMetadata


class TorchPagedAttentionBackend:
    """Slow paged-attention reference backend.

    This backend mirrors the vLLM-style runtime contract used by KVCore's
    Triton path: write projected K/V into the shared paged KV tensor using
    slot mapping, then read historical K/V through the request block table.
    It is intentionally simple and loop-based so correctness tests have a
    readable oracle.
    """

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
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not is_causal:
            raise NotImplementedError(
                "Torch paged attention only supports decoder causal attention"
            )
        if layer_idx is None:
            raise ValueError("layer_idx is required for torch paged attention")
        metadata = self._require_metadata(attn_metadata)
        query_states, key_states, value_states, direct_flat_input = self._canonicalize_qkv(
            query,
            key,
            value,
        )

        num_tokens, num_query_heads, head_dim = query_states.shape
        if num_tokens != metadata.num_scheduled_tokens:
            raise ValueError(
                "metadata.num_scheduled_tokens must match query tokens, "
                f"got {metadata.num_scheduled_tokens} and {num_tokens}"
            )
        if key_states.shape[1] != num_kv_heads or value_states.shape[1] != num_kv_heads:
            raise ValueError(
                "Unexpected KV head count: "
                f"expected {num_kv_heads}, got key={key_states.shape[1]} "
                f"value={value_states.shape[1]}"
            )
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_kv_heads, "
                f"got {num_query_heads} and {num_kv_heads}"
            )

        key_cache = metadata.kv_cache_tensor[0]
        value_cache = metadata.kv_cache_tensor[1]
        block_table = metadata.block_tables[layer_idx].get_device_tensor(metadata.num_reqs)
        block_indices = metadata.block_tables[layer_idx].get_block_indices_device_tensor(
            metadata.num_reqs
        )
        block_size = key_cache.shape[1]

        self._paged_write(
            key_states=key_states,
            value_states=value_states,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=metadata.slot_mapping[layer_idx][:num_tokens],
            block_size=block_size,
        )

        if output is None:
            output_states = torch.empty_like(query_states)
        else:
            if not direct_flat_input:
                raise ValueError("output can only be provided with rank-3 flattened q/k/v inputs")
            output_states = output
            if output_states.shape != query_states.shape:
                raise ValueError(
                    "output shape must match flattened query states, "
                    f"got {tuple(output_states.shape)} and {tuple(query_states.shape)}"
                )
        num_queries_per_kv = num_query_heads // num_kv_heads
        for token_idx in range(num_tokens):
            request_idx = int(metadata.token_request_indices[token_idx].item())
            query_position = int(metadata.flat_positions[token_idx].item())
            for query_head_idx in range(num_query_heads):
                kv_head_idx = query_head_idx // num_queries_per_kv
                keys, values = self._gather_prefix_kv(
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=block_table,
                    block_indices=block_indices,
                    request_idx=request_idx,
                    query_position=query_position,
                    kv_head_idx=kv_head_idx,
                    block_size=block_size,
                )
                scores = (
                    torch.matmul(
                        keys.to(torch.float32),
                        query_states[token_idx, query_head_idx].to(torch.float32),
                    )
                    * scaling
                )
                probs = torch.softmax(scores, dim=0)
                output_states[token_idx, query_head_idx] = torch.matmul(
                    probs,
                    values.to(torch.float32),
                ).to(output_states.dtype)

        if direct_flat_input:
            return output_states
        return output_states.transpose(0, 1).unsqueeze(0)

    @staticmethod
    def _canonicalize_qkv(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        if query.dim() == 3:
            return query.contiguous(), key.contiguous(), value.contiguous(), True
        if query.dim() == 4 and query.shape[0] == 1:
            query_states = query[0].transpose(0, 1).contiguous()
            key_states = key[0].transpose(0, 1).contiguous()
            value_states = value[0].transpose(0, 1).contiguous()
            return query_states, key_states, value_states, False
        raise ValueError(
            "Torch paged attention expects flattened rank-3 q/k/v or legacy "
            f"rank-4 q/k/v with batch size 1, got query shape {tuple(query.shape)}"
        )

    @staticmethod
    def _require_metadata(attn_metadata: object | None) -> PagedAttentionMetadata:
        if not isinstance(attn_metadata, PagedAttentionMetadata):
            raise TypeError(
                "Torch paged attention requires PagedAttentionMetadata, "
                f"got {type(attn_metadata)!r}"
            )
        return attn_metadata

    @staticmethod
    def _paged_write(
        *,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        for token_idx, slot in enumerate(slot_mapping.tolist()):
            if slot < 0:
                continue
            block_id = slot // block_size
            block_offset = slot % block_size
            key_cache[block_id, block_offset].copy_(key_states[token_idx])
            value_cache[block_id, block_offset].copy_(value_states[token_idx])

    @staticmethod
    def _gather_prefix_kv(
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        block_indices: torch.Tensor,
        request_idx: int,
        query_position: int,
        kv_head_idx: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(query_position + 1, device=key_cache.device)
        logical_block_indices = positions // block_size
        block_offsets = positions % block_size
        read_logical_indices = block_indices[request_idx]
        read_block_ids = block_table[request_idx]
        valid_read_blocks = read_logical_indices >= 0
        read_logical_indices = read_logical_indices[valid_read_blocks]
        read_block_ids = read_block_ids[valid_read_blocks].long()
        matches = logical_block_indices[:, None] == read_logical_indices[None, :]
        visible_positions = matches.any(dim=1)
        if not bool(visible_positions.any()):
            return (
                key_cache.new_zeros((0, key_cache.shape[-1])),
                value_cache.new_zeros((0, value_cache.shape[-1])),
            )
        compact_indices = matches.to(torch.int64).argmax(dim=1)
        compact_indices = compact_indices[visible_positions]
        block_ids = read_block_ids.index_select(0, compact_indices)
        block_offsets = block_offsets[visible_positions]
        keys = key_cache[block_ids, block_offsets, kv_head_idx]
        values = value_cache[block_ids, block_offsets, kv_head_idx]
        return keys, values


__all__ = [
    "TorchPagedAttentionBackend",
]
