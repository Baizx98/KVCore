from __future__ import annotations

from torch import nn
from transformers import PretrainedConfig

from kvcore.model.model_loader import DefaultModelLoader, ModelLoadConfig


class ModelRunner:
    """Owns model creation and weight loading for the current process.

    This intentionally only covers the vLLM-like early lifecycle:
    create model skeleton -> load weights.
    Runtime concerns such as profiling, KV cache allocation, and execution
    loops are deferred to later work.
    """

    def __init__(self, load_config: ModelLoadConfig) -> None:
        self.load_config = load_config
        self.model_loader = DefaultModelLoader(load_config)
        self.hf_config: PretrainedConfig | None = None
        self.model: nn.Module | None = None

    def create_model(self) -> nn.Module:
        self.hf_config = self.model_loader.load_config_from_source()
        self.model = self.model_loader.create_model(self.hf_config)
        return self.model

    def load_model(self) -> nn.Module:
        self.model = self.model_loader.load_model()
        self.hf_config = getattr(self.model, "config", None)
        return self.model
