from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model: str | Path
    tokenizer: str | Path | None = None
    trust_remote_code: bool = False
    hf_config: PretrainedConfig | None = None
    dtype: torch.dtype | None = None
    max_model_len: int | None = None
    attn_backend: str | None = None

    def __post_init__(self) -> None:
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {self.max_model_len}")

    def with_hf_config(self, hf_config: PretrainedConfig) -> ModelConfig:
        return replace(self, hf_config=hf_config)


@dataclass(frozen=True, slots=True)
class LoadConfig:
    revision: str | None = None
    load_format: str = "auto"
    download_dir: str | None = None
    ignore_patterns: list[str] = field(default_factory=list)
    local_files_only: bool = False
    strict: bool = True


@dataclass(frozen=True, slots=True)
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: int | None = None
    profile_kv_cache: bool = True
    gpu_memory_utilization: float = 0.9
    enable_caching: bool = True

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.num_gpu_blocks is not None and self.num_gpu_blocks <= 0:
            raise ValueError(f"num_gpu_blocks must be positive, got {self.num_gpu_blocks}")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                "gpu_memory_utilization must be in (0, 1], "
                f"got {self.gpu_memory_utilization}"
            )


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_num_seqs: int = 8
    max_num_scheduled_tokens: int = 512
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    long_prefill_token_threshold: int = 0

    def __post_init__(self) -> None:
        if self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {self.max_num_seqs}")
        if self.max_num_scheduled_tokens <= 0:
            raise ValueError(
                "max_num_scheduled_tokens must be positive, "
                f"got {self.max_num_scheduled_tokens}"
            )
        if self.max_num_partial_prefills <= 0:
            raise ValueError(
                "max_num_partial_prefills must be positive, "
                f"got {self.max_num_partial_prefills}"
            )
        if self.max_long_partial_prefills <= 0:
            raise ValueError(
                "max_long_partial_prefills must be positive, "
                f"got {self.max_long_partial_prefills}"
            )
        if self.long_prefill_token_threshold < 0:
            raise ValueError(
                "long_prefill_token_threshold must be non-negative, "
                f"got {self.long_prefill_token_threshold}"
            )


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    device: str | None = None


@dataclass(frozen=True, slots=True)
class KVCoreConfig:
    model_config: ModelConfig
    load_config: LoadConfig = field(default_factory=LoadConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)

    def __post_init__(self) -> None:
        if self.model_config.attn_backend is not None:
            return
        object.__setattr__(
            self,
            "model_config",
            replace(
                self.model_config,
                attn_backend=self._default_attn_backend(self.device_config.device),
            ),
        )

    @staticmethod
    def _default_attn_backend(device: str | None) -> str:
        if (device is None and torch.cuda.is_available()) or (
            device is not None and torch.device(device).type == "cuda"
        ):
            return "triton_paged"
        return "torch_paged"


__all__ = [
    "CacheConfig",
    "DeviceConfig",
    "KVCoreConfig",
    "LoadConfig",
    "ModelConfig",
    "SchedulerConfig",
]
