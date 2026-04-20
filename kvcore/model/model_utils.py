from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def load_named_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    tied_lm_head: bool,
) -> set[str]:
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    loaded_params: set[str] = set()

    for name, loaded_weight in weights:
        if name.endswith("rotary_emb.inv_freq"):
            continue
        if name == "lm_head.weight" and tied_lm_head:
            loaded_params.add(name)
            continue

        if name in params:
            param = params[name]
            param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))
            loaded_params.add(name)
            continue

        if name in buffers:
            buffer = buffers[name]
            buffer.data.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
            loaded_params.add(name)

    return loaded_params


def prepare_model_inputs(
    module: nn.Module,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
) -> torch.Tensor:
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is not None:
        return inputs_embeds
    return module.embed_input_ids(input_ids)


def validate_kv_caches(
    kv_caches: list[object | None] | None,
    num_layers: int,
) -> None:
    if kv_caches is not None and len(kv_caches) != num_layers:
        raise ValueError(
            "kv_caches must match the number of decoder layers, "
            f"got {len(kv_caches)} caches for {num_layers} layers"
        )


def infer_batch_and_seq_len(hidden_states: torch.Tensor) -> tuple[int, int]:
    if hidden_states.dim() == 2:
        return 1, hidden_states.size(0)
    if hidden_states.dim() == 3:
        return hidden_states.size(0), hidden_states.size(1)
    raise ValueError(
        "hidden_states must be rank-2 or rank-3, "
        f"got shape {tuple(hidden_states.shape)}"
    )


def apply_sliding_window_metadata(
    attn_metadata: object | None,
    sliding_window: int | None,
) -> object | None:
    if sliding_window is None:
        return attn_metadata
    if attn_metadata is None:
        return {"sliding_window": sliding_window}
    if isinstance(attn_metadata, dict):
        merged = dict(attn_metadata)
        merged.setdefault("sliding_window", sliding_window)
        return merged
    return attn_metadata
