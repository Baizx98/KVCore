from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from kvcore.kv.sparse import SparseKVMode, SparseKVSelectionInterval

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
class SparseKVConfig:
    mode: str = SparseKVMode.DISABLED.value
    selection_interval: str = SparseKVSelectionInterval.STEP.value
    selection_interval_tokens: int | None = None
    compression_ratio: float = 0.5
    q_window_size: int = 32
    prefix_sink_blocks: int = 1
    protected_recent_blocks: int = 2
    score_ema_alpha: float = 0.8
    summary_topk_keys: int = 4
    mean_key_weight: float = 0.75
    enable_prefill_sparsity: bool = False
    enable_decode_sparsity: bool = True

    def __post_init__(self) -> None:
        valid_modes = {mode.value for mode in SparseKVMode}
        if self.mode not in valid_modes:
            raise ValueError(f"sparse KV mode must be one of {valid_modes}, got {self.mode}")
        valid_intervals = {interval.value for interval in SparseKVSelectionInterval}
        if self.selection_interval not in valid_intervals:
            raise ValueError(
                "sparse KV selection_interval must be one of "
                f"{valid_intervals}, got {self.selection_interval}"
            )
        if self.selection_interval == SparseKVSelectionInterval.N_TOKENS.value:
            if self.selection_interval_tokens is None or self.selection_interval_tokens <= 0:
                raise ValueError(
                    "selection_interval_tokens must be positive when "
                    "selection_interval is n_tokens"
                )
        if not 0 <= self.compression_ratio < 1:
            raise ValueError(
                f"compression_ratio must be in [0, 1), got {self.compression_ratio}"
            )
        if self.q_window_size <= 0:
            raise ValueError(f"q_window_size must be positive, got {self.q_window_size}")
        if self.prefix_sink_blocks < 0:
            raise ValueError(
                f"prefix_sink_blocks must be non-negative, got {self.prefix_sink_blocks}"
            )
        if self.protected_recent_blocks < 0:
            raise ValueError(
                "protected_recent_blocks must be non-negative, "
                f"got {self.protected_recent_blocks}"
            )
        if not 0 <= self.score_ema_alpha <= 1:
            raise ValueError(
                f"score_ema_alpha must be in [0, 1], got {self.score_ema_alpha}"
            )
        if self.summary_topk_keys <= 0:
            raise ValueError(
                f"summary_topk_keys must be positive, got {self.summary_topk_keys}"
            )
        if not 0 <= self.mean_key_weight <= 1:
            raise ValueError(
                f"mean_key_weight must be in [0, 1], got {self.mean_key_weight}"
            )

    @property
    def is_enabled(self) -> bool:
        return self.mode != SparseKVMode.DISABLED.value


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    device: str | None = None


@dataclass(frozen=True, slots=True)
class KVCoreConfig:
    model_config: ModelConfig
    load_config: LoadConfig = field(default_factory=LoadConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    sparse_kv_config: SparseKVConfig = field(default_factory=SparseKVConfig)
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
    "SparseKVConfig",
]
