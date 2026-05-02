from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from kvcore.config import KVCoreConfig, LoadConfig

if TYPE_CHECKING:
    from torch import nn


class BaseModelLoader(ABC):
    def __init__(self, kvcore_config: KVCoreConfig) -> None:
        self.kvcore_config = kvcore_config
        self.model_config = kvcore_config.model_config
        self.load_config = kvcore_config.load_config
        self.device_config = kvcore_config.device_config

    @abstractmethod
    def load_model(self) -> nn.Module:
        raise NotImplementedError


__all__ = ["BaseModelLoader", "LoadConfig"]
