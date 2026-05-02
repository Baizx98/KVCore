from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig

from kvcore.config import KVCoreConfig
from kvcore.model.models import (
    Llama3ForCausalLM,
    MistralForCausalLM,
    Qwen3ForCausalLM,
)
from kvcore.utils.log import get_logger

from .base_loader import BaseModelLoader
from .utils import resolve_model_dtype, resolve_weight_tensor_device, set_default_torch_dtype
from .weight_utils import (
    filter_duplicate_safetensors_files,
    get_weight_files_from_index,
    pt_weights_iterator,
    safetensors_weights_iterator,
)

logger = get_logger(__name__)

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "llama": Llama3ForCausalLM,
    "mistral": MistralForCausalLM,
    "qwen3": Qwen3ForCausalLM,
}

SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"
PT_INDEX_NAME = "pytorch_model.bin.index.json"
CONSOLIDATED_INDEX_NAME = "consolidated.safetensors.index.json"


class DefaultModelLoader(BaseModelLoader):
    def load_model(self) -> nn.Module:
        hf_config = self._load_config()
        model = self._initialize_model(hf_config)
        loaded_weights = self.load_weights(model)
        if self.load_config.strict:
            self._validate_loaded_weights(model, loaded_weights)
        if self.device_config.device is not None:
            model = model.to(self.device_config.device)
        model.eval()
        return model

    def _load_config(self) -> PretrainedConfig:
        return AutoConfig.from_pretrained(
            self.model_config.model,
            revision=self.load_config.revision,
            cache_dir=self.load_config.download_dir,
            trust_remote_code=self.model_config.trust_remote_code,
            local_files_only=self.load_config.local_files_only,
        )

    def _initialize_model(self, hf_config: PretrainedConfig) -> nn.Module:
        model_cls = MODEL_REGISTRY.get(hf_config.model_type)
        if model_cls is None:
            raise ValueError(
                f"Unsupported model_type: {hf_config.model_type!r}. "
                f"Supported: {sorted(MODEL_REGISTRY)}"
            )

        dtype = self.model_config.dtype or resolve_model_dtype(hf_config)
        with set_default_torch_dtype(dtype):
            model_config = self.model_config.with_hf_config(hf_config)
            return model_cls(
                kvcore_config=KVCoreConfig(
                    model_config=model_config,
                    load_config=self.load_config,
                    cache_config=self.kvcore_config.cache_config,
                    scheduler_config=self.kvcore_config.scheduler_config,
                    device_config=self.device_config,
                )
            )

    def load_weights(self, model: nn.Module) -> set[str]:
        load_weights = getattr(model, "load_weights", None)
        if not callable(load_weights):
            raise TypeError(f"{type(model).__name__} must implement load_weights()")
        load_weights = cast(
            Callable[[Iterable[tuple[str, torch.Tensor]]], set[str]],
            load_weights,
        )

        model_path = self._resolve_model_path()
        weight_files = self._prepare_weights(model_path)
        return load_weights(self._get_weights_iterator(weight_files))

    def _resolve_model_path(self) -> Path:
        model_path = Path(self.model_config.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_config.model}")
        return model_path.resolve()

    def _prepare_weights(self, model_path: Path) -> list[Path]:
        load_format = self._resolve_load_format(model_path)

        if load_format in {"hf", "safetensors", "mistral"}:
            index_name = (
                CONSOLIDATED_INDEX_NAME if load_format == "mistral" else SAFETENSORS_INDEX_NAME
            )
            indexed = get_weight_files_from_index(model_path, index_name)
            if indexed:
                return indexed

        if load_format in {"hf", "bin", "pt"}:
            indexed = get_weight_files_from_index(model_path, PT_INDEX_NAME)
            if indexed:
                return indexed

        patterns = self._allow_patterns_for_load_format(load_format)
        weight_files = self._find_weight_files(model_path, patterns)
        if load_format in {"hf", "safetensors", "mistral"}:
            weight_files = filter_duplicate_safetensors_files(weight_files, model_path)
        if not weight_files:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_path}` using load_format={load_format}"
            )
        return weight_files

    def _resolve_load_format(self, model_path: Path) -> str:
        load_format = self.load_config.load_format
        if load_format != "auto":
            return load_format
        if list(model_path.glob("consolidated*.safetensors")):
            return "mistral"
        if list(model_path.glob("*.safetensors")):
            return "safetensors"
        if list(model_path.glob("*.bin")):
            return "bin"
        if list(model_path.glob("*.pt")):
            return "pt"
        return "hf"

    @staticmethod
    def _allow_patterns_for_load_format(load_format: str) -> list[str]:
        if load_format == "hf":
            return ["*.safetensors", "*.bin", "*.pt"]
        if load_format == "safetensors":
            return ["*.safetensors"]
        if load_format == "mistral":
            return ["consolidated*.safetensors"]
        if load_format == "bin":
            return ["*.bin"]
        if load_format == "pt":
            return ["*.pt"]
        raise ValueError(f"Unknown load_format: {load_format}")

    @staticmethod
    def _find_weight_files(model_path: Path, patterns: list[str]) -> list[Path]:
        for pattern in patterns:
            matched_files = sorted(
                path
                for path in model_path.iterdir()
                if path.is_file() and fnmatch.fnmatch(path.name, pattern)
            )
            if matched_files:
                return matched_files
        return []

    def _get_weights_iterator(self, weight_files: list[Path]) -> Iterable[tuple[str, torch.Tensor]]:
        if not weight_files:
            return iter(())

        tensor_device = resolve_weight_tensor_device(self.device_config.device)
        if all(path.suffix == ".safetensors" for path in weight_files):
            return safetensors_weights_iterator(weight_files, device=tensor_device)
        return pt_weights_iterator(weight_files, device=tensor_device)

    @staticmethod
    def _validate_loaded_weights(model: nn.Module, loaded_weights: set[str]) -> None:
        expected_weights = {
            name
            for name in model.state_dict()
            if not name.endswith("rotary_emb.inv_freq")
        }
        tie_word_embeddings = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
        if tie_word_embeddings:
            expected_weights.discard("lm_head.weight")

        missing = sorted(expected_weights - loaded_weights)
        if missing:
            raise ValueError(
                "Missing weights after load: "
                + ", ".join(missing[:20])
                + (" ..." if len(missing) > 20 else "")
            )


def get_model_loader(kvcore_config: KVCoreConfig) -> BaseModelLoader:
    return DefaultModelLoader(kvcore_config)


def get_model(kvcore_config: KVCoreConfig) -> nn.Module:
    return get_model_loader(kvcore_config).load_model()


__all__ = ["DefaultModelLoader", "get_model_loader", "get_model", "MODEL_REGISTRY"]
