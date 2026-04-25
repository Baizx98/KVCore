from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from kvcore.sched.utils import SchedulerConfig

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

    from kvcore.model.model_loader import ModelLoadConfig


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model: str
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    load_format: str = "auto"
    download_dir: str | None = None
    ignore_patterns: tuple[str, ...] = ()
    attn_backend: str | None = None
    device: str | None = None
    hf_config: PretrainedConfig | None = None

    @classmethod
    def from_load_config(cls, load_config: ModelLoadConfig) -> ModelConfig:
        return cls(
            model=load_config.model,
            revision=load_config.revision,
            trust_remote_code=load_config.trust_remote_code,
            local_files_only=load_config.local_files_only,
            load_format=load_config.load_format,
            download_dir=load_config.download_dir,
            ignore_patterns=tuple(load_config.ignore_patterns),
            attn_backend=load_config.attn_backend,
            device=load_config.device,
        )

    def with_hf_config(self, hf_config: PretrainedConfig) -> ModelConfig:
        config = replace(self, hf_config=hf_config)
        return config

    def to_load_config(self) -> ModelLoadConfig:
        from kvcore.model.model_loader import ModelLoadConfig

        return ModelLoadConfig(
            model=self.model,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            load_format=self.load_format,
            download_dir=self.download_dir,
            ignore_patterns=list(self.ignore_patterns),
            attn_backend=self.attn_backend,
            device=self.device,
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    block_size: int = 16
    num_gpu_blocks: int | None = 2048
    max_model_len: int | None = None
    profile_kv_cache: bool = False
    gpu_memory_utilization: float = 0.9

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
class KVCoreConfig:
    model: ModelConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


__all__ = [
    "KVCoreConfig",
    "ModelConfig",
    "RuntimeConfig",
]
