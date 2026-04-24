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
    ) -> torch.Tensor:
        if not is_causal:
            raise NotImplementedError(
                "Torch paged attention only supports decoder causal attention"
            )
        if layer_idx is None:
            raise ValueError("layer_idx is required for torch paged attention")
        metadata = self._require_metadata(attn_metadata)
        if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
            raise ValueError(
                "Torch paged attention expects flattened runtime inputs with batch size 1, "
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
        block_size = key_cache.shape[1]

        self._paged_write(
            key_states=key_states,
            value_states=value_states,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=metadata.slot_mapping[layer_idx][:num_tokens],
            block_size=block_size,
        )

        output = torch.empty_like(query_states)
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
                output[token_idx, query_head_idx] = torch.matmul(
                    probs,
                    values.to(torch.float32),
                ).to(output.dtype)

        return output.transpose(0, 1).unsqueeze(0)

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
        request_idx: int,
        query_position: int,
        kv_head_idx: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(query_position + 1, device=key_cache.device)
        block_indices = positions // block_size
        block_offsets = positions % block_size
        block_ids = block_table[request_idx, block_indices].long()
        keys = key_cache[block_ids, block_offsets, kv_head_idx]
        values = value_cache[block_ids, block_offsets, kv_head_idx]
        return keys, values


__all__ = [
    "TorchPagedAttentionBackend",
]
