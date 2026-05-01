from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn


@dataclass(slots=True)
class LoadConfig:
    model: str | Path
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    load_format: str = "auto"
    download_dir: str | None = None
    ignore_patterns: list[str] = field(default_factory=list)
    attn_backend: str | None = None
    device: str | None = None
    dtype: torch.dtype | None = None
    strict: bool = True

class BaseModelLoader(ABC):
    def __init__(self, load_config: LoadConfig) -> None:
        self.load_config = load_config

    @abstractmethod
    def load_model(self) -> nn.Module:
        raise NotImplementedError

    def download_model(self) -> None:
        pass


__all__ = ["BaseModelLoader", "LoadConfig"]
