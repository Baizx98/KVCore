from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _repeat_kv(hidden_states: torch.Tensor, num_attention_heads: int) -> torch.Tensor:
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if num_kv_heads == num_attention_heads:
        return hidden_states
    if num_attention_heads % num_kv_heads != 0:
        raise ValueError(
            "num_attention_heads must be divisible by num_kv_heads, "
            f"got {num_attention_heads=} and {num_kv_heads=}"
        )
    repeat_factor = num_attention_heads // num_kv_heads
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_kv_heads,
        repeat_factor,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch_size, num_attention_heads, seq_len, head_dim)


def _maybe_update_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    kv_cache: object | None,
    attn_metadata: object | None,
    layer_idx: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_cache is None or not hasattr(kv_cache, "update"):
        return key, value

    update = kv_cache.update
    for kwargs in (
        {"key": key, "value": value, "attn_metadata": attn_metadata, "layer_idx": layer_idx},
        {"key": key, "value": value, "layer_idx": layer_idx},
        {"key": key, "value": value},
    ):
        try:
            updated = update(**kwargs)
        except TypeError:
            continue
        if not isinstance(updated, tuple) or len(updated) != 2:
            raise TypeError("kv_cache.update must return a (key, value) tuple")
        return updated

    updated = update(key, value)
    if not isinstance(updated, tuple) or len(updated) != 2:
        raise TypeError("kv_cache.update must return a (key, value) tuple")
    return updated


def _extract_metadata_value(attn_metadata: object | None, key: str) -> Any:
    if attn_metadata is None:
        return None
    if isinstance(attn_metadata, dict):
        return attn_metadata.get(key)
    return getattr(attn_metadata, key, None)


class TorchSDPAAttentionBackend:
    """Minimal eager backend used to keep the Llama model executable."""

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
        kv_cache: object | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        key, value = _maybe_update_kv_cache(
            key,
            value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            layer_idx=layer_idx,
        )

        num_attention_heads = query.size(1)
        if key.size(1) != num_kv_heads or value.size(1) != num_kv_heads:
            raise ValueError(
                "Unexpected KV head count returned by attention projections/cache update: "
                f"expected {num_kv_heads}, got key={key.size(1)} value={value.size(1)}"
            )

        key = _repeat_kv(key, num_attention_heads)
        value = _repeat_kv(value, num_attention_heads)

        attn_mask = _extract_metadata_value(attn_metadata, "attention_mask")
        metadata_is_causal = _extract_metadata_value(attn_metadata, "is_causal")
        effective_is_causal = is_causal if metadata_is_causal is None else bool(metadata_is_causal)

        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=effective_is_causal,
            scale=scaling,
        )

