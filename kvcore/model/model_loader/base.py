from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from torch import nn
from transformers import PretrainedConfig


@dataclass(slots=True)
class ModelLoadConfig:
    model: str
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    load_format: str = "auto"
    download_dir: str | None = None
    ignore_patterns: list[str] = field(default_factory=list)
    attn_backend: str | None = None
    device: str | None = None


class BaseModelLoader:
    def __init__(self, load_config: ModelLoadConfig) -> None:
        self.load_config = load_config

    def load_config_from_source(self) -> PretrainedConfig:
        raise NotImplementedError

    def create_model(self, hf_config: PretrainedConfig) -> nn.Module:
        raise NotImplementedError

    def resolve_model_path(self) -> Path:
        raise NotImplementedError

    def load_weights(self, model: nn.Module, model_path: Path) -> set[str]:
        raise NotImplementedError

    def load_model(self) -> nn.Module:
        hf_config = self.load_config_from_source()
        model = self.create_model(hf_config)
        model_path = self.resolve_model_path()
        self.load_weights(model, model_path)
        if self.load_config.device is not None:
            model = model.to(self.load_config.device)
        model.eval()
        return model
