from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import torch
from transformers.configuration_utils import PretrainedConfig


def resolve_model_dtype(hf_config: PretrainedConfig) -> torch.dtype | None:
    dtype = getattr(hf_config, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype, None)
    return None


@contextmanager
def set_default_torch_dtype(dtype: torch.dtype | None) -> Generator[None, None, None]:
    if dtype is None or not dtype.is_floating_point:
        yield
        return

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)


def resolve_weight_tensor_device(config_device: str | None) -> str:
    if config_device is None:
        return "cpu"
    device = torch.device(config_device)
    if device.type != "cuda":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return str(device)


__all__ = [
    "resolve_model_dtype",
    "set_default_torch_dtype",
    "resolve_weight_tensor_device",
]
