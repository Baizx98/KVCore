from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from kvcore.config import KVCoreConfig

_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def maybe_prefix(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}.{suffix}"


def extract_layer_index(prefix: str) -> int:
    matches = list(_LAYER_INDEX_PATTERN.finditer(prefix))
    if not matches:
        raise ValueError(f"Unable to extract layer index from prefix: {prefix!r}")
    return int(matches[-1].group(1))


def get_hf_config(kvcore_config: KVCoreConfig) -> PretrainedConfig:
    hf_config = kvcore_config.model_config.hf_config
    if hf_config is None:
        raise ValueError("KVCoreConfig.model_config.hf_config must be set before model creation")
    return hf_config


def load_named_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    tied_lm_head: bool,
    stacked_params_mapping: tuple[tuple[str, str, str | int], ...] = (),
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

        stacked_name = _load_stacked_weight(
            params=params,
            name=name,
            loaded_weight=loaded_weight,
            stacked_params_mapping=stacked_params_mapping,
        )
        if stacked_name is not None:
            loaded_params.add(stacked_name)
            continue

        if name in params:
            param = params[name]
            _copy_weight_into_tensor(loaded_weight, param.data)
            loaded_params.add(name)
            continue

        if name in buffers:
            buffer = buffers[name]
            _copy_weight_into_tensor(loaded_weight, buffer.data)
            loaded_params.add(name)

    return loaded_params


def _load_stacked_weight(
    *,
    params: dict[str, nn.Parameter],
    name: str,
    loaded_weight: torch.Tensor,
    stacked_params_mapping: tuple[tuple[str, str, str | int], ...],
) -> str | None:
    for param_name, weight_name, shard_id in stacked_params_mapping:
        if weight_name not in name:
            continue
        stacked_name = name.replace(weight_name, param_name)
        if stacked_name not in params:
            return None
        param = params[stacked_name]
        loaded_weight = _prepare_weight_for_tensor(loaded_weight, param.data)
        offset = _stacked_weight_offset(param, loaded_weight, shard_id)
        param.data.narrow(0, offset, loaded_weight.shape[0]).copy_(loaded_weight)
        return stacked_name
    return None


def _copy_weight_into_tensor(loaded_weight: torch.Tensor, target: torch.Tensor) -> None:
    if loaded_weight.device == target.device and loaded_weight.dtype == target.dtype:
        target.copy_(loaded_weight)
        return
    target.copy_(loaded_weight.to(device=target.device, dtype=target.dtype))


def _prepare_weight_for_tensor(
    loaded_weight: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if loaded_weight.device == target.device and loaded_weight.dtype == target.dtype:
        return loaded_weight
    return loaded_weight.to(device=target.device, dtype=target.dtype)


def _stacked_weight_offset(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str | int,
) -> int:
    shard_size = loaded_weight.shape[0]
    if shard_id == "q":
        return 0
    if shard_id == "k":
        return param.shape[0] - (2 * shard_size)
    if shard_id == "v":
        return param.shape[0] - shard_size
    return int(shard_id) * shard_size


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
