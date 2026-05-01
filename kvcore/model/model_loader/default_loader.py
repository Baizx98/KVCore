from __future__ import annotations

import concurrent.futures
import dataclasses
import fnmatch
import json
import os
import time
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig

from kvcore.config import KVCoreConfig, ModelConfig
from kvcore.model.model_loader.base import BaseModelLoader
from kvcore.model.models import (
    Llama3ForCausalLM,
    MistralForCausalLM,
    Qwen3ForCausalLM,
)
from kvcore.utils.log import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "llama": Llama3ForCausalLM,
    "mistral": MistralForCausalLM,
    "qwen3": Qwen3ForCausalLM,
}

SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
PT_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
DEFAULT_WEIGHT_LOAD_THREADS = 4


class DefaultModelLoader(BaseModelLoader):
    """Default loader for local or Hugging Face model weights.

    This keeps the same conceptual split as vLLM's default loader:
    resolve model source -> choose files by load format -> iterate tensors ->
    call the model's `load_weights`.
    """

    @dataclasses.dataclass(slots=True)
    class Source:
        model_or_path: str
        revision: str | None
        prefix: str = ""
        fall_back_to_pt: bool = True
        allow_patterns_overrides: list[str] | None = None

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def load_config_from_source(self) -> PretrainedConfig:
        return AutoConfig.from_pretrained(
            self.load_config.model,
            revision=self.load_config.revision,
            trust_remote_code=self.load_config.trust_remote_code,
            local_files_only=self.load_config.local_files_only,
        )

    def create_model(self, hf_config: PretrainedConfig) -> nn.Module:
        model_cls = MODEL_REGISTRY.get(hf_config.model_type)
        if model_cls is None:
            raise ValueError(
                f"Unsupported model_type={hf_config.model_type!r}. "
                f"Supported model types: {sorted(MODEL_REGISTRY)}"
            )
        with _temporary_default_dtype(_resolve_model_dtype(hf_config)):
            kvcore_config = KVCoreConfig(
                model=ModelConfig.from_load_config(self.load_config).with_hf_config(hf_config)
            )
            model = model_cls(kvcore_config=kvcore_config)
            return model

    def resolve_model_path(self) -> Path:
        candidate = Path(self.load_config.model)
        if candidate.exists():
            return candidate.resolve()

        snapshot_path = snapshot_download(
            repo_id=self.load_config.model,
            revision=self.load_config.revision,
            local_files_only=self.load_config.local_files_only,
            cache_dir=self.load_config.download_dir,
            ignore_patterns=self.load_config.ignore_patterns or None,
        )
        return Path(snapshot_path)

    def load_model(self) -> nn.Module:
        hf_config = self.load_config_from_source()
        model = self.create_model(hf_config)
        self.load_weights(model, self.resolve_model_path())
        if self.load_config.device is not None:
            model = model.to(self.load_config.device)
        model.eval()
        return model

    def load_weights(self, model: nn.Module, model_path: Path) -> set[str]:
        loaded_params = self._load_model_weights(model, self.get_all_weights(model_path, model))
        self._validate_loaded_weights(model, loaded_params)
        if self.counter_before_loading_weights and self.counter_after_loading_weights:
            logger.info(
                "Model weights loaded params=%d elapsed=%.3fs",
                len(loaded_params),
                self.counter_after_loading_weights - self.counter_before_loading_weights,
            )
        return loaded_params

    def get_all_weights(
        self,
        model_path: Path,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        primary_weights = DefaultModelLoader.Source(
            model_or_path=str(model_path),
            revision=self.load_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        yield from self._get_weights_iterator(primary_weights, model)

        secondary_weights = getattr(model, "secondary_weights", ())
        for source in secondary_weights:
            yield from self._get_weights_iterator(source, model)

    def download_model(self) -> Path:
        return self.resolve_model_path()

    def _get_weights_iterator(
        self,
        source: Source,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        model_path, weight_files = self._prepare_weights(
            source.model_or_path,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()

        tensor_device = self._resolve_weight_tensor_device(model)
        if self._should_use_multithread_load(weight_files):
            filepaths = [model_path / filename for filename in weight_files]
            if all(filepath.suffix == ".safetensors" for filepath in filepaths):
                yield from self._iter_safetensors_files_multithread(filepaths, source.prefix)
                return
            if all(filepath.suffix != ".safetensors" for filepath in filepaths):
                yield from self._iter_pt_files_multithread(filepaths, source.prefix)
                return

        for filename in weight_files:
            filepath = model_path / filename
            if filepath.suffix == ".safetensors":
                yield from self._iter_safetensors_file(filepath, source.prefix, tensor_device)
                continue

            state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
            yield from ((source.prefix + name, tensor) for name, tensor in state_dict.items())

    def _resolve_weight_tensor_device(self, model: nn.Module) -> str:
        device = self.load_config.device
        if device is None:
            return "cpu"
        try:
            current_device = next(model.parameters()).device
        except StopIteration:
            return "cpu"
        if current_device.type == "cpu":
            return "cpu"
        torch_device = torch.device(device)
        if torch_device.type != "cuda":
            return "cpu"
        if not torch.cuda.is_available():
            return "cpu"
        return str(torch_device)

    def _should_use_multithread_load(self, weight_files: list[str]) -> bool:
        if len(weight_files) <= 1:
            return False
        disabled = os.environ.get("KVCORE_DISABLE_MULTITHREAD_WEIGHT_LOAD")
        if disabled is not None and disabled.lower() in {"1", "true", "yes"}:
            return False
        return self._num_weight_load_threads() > 1

    @staticmethod
    def _num_weight_load_threads() -> int:
        raw_value = os.environ.get("KVCORE_WEIGHT_LOAD_THREADS")
        if raw_value is None:
            return DEFAULT_WEIGHT_LOAD_THREADS
        try:
            return max(1, int(raw_value))
        except ValueError:
            logger.warning(
                "Ignoring invalid KVCORE_WEIGHT_LOAD_THREADS=%r; using %d",
                raw_value,
                DEFAULT_WEIGHT_LOAD_THREADS,
            )
            return DEFAULT_WEIGHT_LOAD_THREADS

    def _iter_safetensors_file(
        self,
        filepath: Path,
        prefix: str,
        device: str,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        try:
            yield from self._iter_safetensors_file_on_device(filepath, prefix, device)
        except RuntimeError as exc:
            if device == "cpu" or not self._is_cuda_memory_error(exc):
                raise
            logger.warning(
                "Falling back to CPU weight loading after CUDA allocation failure "
                "file=%s error=%s",
                filepath.name,
                exc,
            )
            torch.cuda.empty_cache()
            yield from self._iter_safetensors_file_on_device(filepath, prefix, "cpu")

    def _iter_safetensors_files_multithread(
        self,
        filepaths: list[Path],
        prefix: str,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        logger.info(
            "Loading safetensors shards with multithread CPU loader files=%d threads=%d",
            len(filepaths),
            self._num_weight_load_threads(),
        )
        for state_dict in self._iter_loaded_state_dicts(
            filepaths,
            loader=lambda filepath: load_safetensors_file(str(filepath), device="cpu"),
        ):
            for name in list(state_dict):
                yield prefix + name, state_dict.pop(name)
            del state_dict

    def _iter_pt_files_multithread(
        self,
        filepaths: list[Path],
        prefix: str,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        logger.info(
            "Loading PyTorch checkpoint shards with multithread CPU loader files=%d threads=%d",
            len(filepaths),
            self._num_weight_load_threads(),
        )
        for state_dict in self._iter_loaded_state_dicts(
            filepaths,
            loader=lambda filepath: torch.load(
                filepath,
                map_location="cpu",
                weights_only=True,
            ),
        ):
            for name in list(state_dict):
                yield prefix + name, state_dict.pop(name)
            del state_dict

    def _iter_loaded_state_dicts(
        self,
        filepaths: list[Path],
        *,
        loader: Callable[[Path], dict[str, torch.Tensor]],
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        max_workers = min(self._num_weight_load_threads(), len(filepaths))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending = {
                executor.submit(loader, filepath): filepath
                for filepath in filepaths[:max_workers]
            }
            next_file_index = max_workers
            while pending:
                done, _pending = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    filepath = pending.pop(future)
                    try:
                        yield future.result()
                    except Exception as exc:
                        raise RuntimeError(f"Failed to load weight shard: {filepath}") from exc
                    if next_file_index < len(filepaths):
                        next_path = filepaths[next_file_index]
                        pending[executor.submit(loader, next_path)] = next_path
                        next_file_index += 1

    @staticmethod
    def _iter_safetensors_file_on_device(
        filepath: Path,
        prefix: str,
        device: str,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        with safe_open(filepath, framework="pt", device=device) as weights_file:
            for name in weights_file.keys():
                yield prefix + name, weights_file.get_tensor(name)

    @staticmethod
    def _is_cuda_memory_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "cuda" in message and (
            "out of memory" in message
            or "allocation" in message
            or "cublas" in message
        )

    def _prepare_weights(
        self,
        model_name_or_path: str,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[Path, list[str]]:
        model_path = Path(model_name_or_path)
        if not model_path.exists():
            model_path = self.resolve_model_path()
        model_path = model_path.resolve()

        load_format = self._resolve_load_format(model_path)
        allow_patterns = self._allow_patterns_for_load_format(load_format)
        if fall_back_to_pt:
            allow_patterns.append("*.pt")
        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        weight_files = self._find_weight_files(model_path, allow_patterns, load_format)
        if not weight_files:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}` "
                f"using load_format={load_format}"
            )
        return model_path, weight_files

    def _resolve_load_format(self, model_path: Path) -> str:
        load_format = self.load_config.load_format
        if load_format == "auto":
            mistral_files = list(model_path.glob("consolidated*.safetensors"))
            return "mistral" if mistral_files else "hf"
        return load_format

    @staticmethod
    def _allow_patterns_for_load_format(load_format: str) -> list[str]:
        if load_format == "hf":
            return ["*.safetensors", "*.bin"]
        if load_format == "safetensors":
            return ["*.safetensors"]
        if load_format == "mistral":
            return ["consolidated*.safetensors"]
        if load_format == "pt":
            return ["*.pt"]
        raise ValueError(f"Unknown load_format: {load_format}")

    def _find_weight_files(
        self,
        model_path: Path,
        allow_patterns: list[str],
        load_format: str,
    ) -> list[str]:
        if load_format in {"hf", "safetensors", "mistral"}:
            index_name = (
                "consolidated.safetensors.index.json"
                if load_format == "mistral"
                else SAFE_WEIGHTS_INDEX_NAME
            )
            index_path = model_path / index_name
            if index_path.exists():
                return self._weight_filenames_from_index(index_path)

        if load_format == "hf":
            pt_index = model_path / PT_WEIGHTS_INDEX_NAME
            if pt_index.exists():
                return self._weight_filenames_from_index(pt_index)

        matched_files: list[str] = []
        for pattern in allow_patterns:
            pattern_matches = sorted(
                path.name
                for path in model_path.iterdir()
                if path.is_file() and fnmatch.fnmatch(path.name, pattern)
            )
            if pattern_matches:
                matched_files.extend(pattern_matches)
                if pattern == "*.safetensors":
                    return self._filter_duplicate_safetensors_files(matched_files, model_path)
                return matched_files
        return matched_files

    @staticmethod
    def _filter_duplicate_safetensors_files(
        filenames: list[str],
        model_path: Path,
    ) -> list[str]:
        index_path = model_path / SAFE_WEIGHTS_INDEX_NAME
        if not index_path.exists():
            return filenames
        indexed = set(DefaultModelLoader._weight_filenames_from_index(index_path))
        filtered = [filename for filename in filenames if filename in indexed]
        return filtered or filenames

    @staticmethod
    def _weight_filenames_from_index(index_path: Path) -> list[str]:
        index_data = json.loads(index_path.read_text())
        weight_map = index_data.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid weight index file: {index_path}")
        return sorted(set(weight_map.values()))

    def _load_model_weights(
        self,
        model: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        if hasattr(model, "load_weights"):
            loaded_params = model.load_weights(weights)
        else:
            state_dict = dict(weights)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise ValueError(f"Unexpected weight names: {unexpected}")
            loaded_params = set(state_dict) - set(missing)

        self.counter_after_loading_weights = time.perf_counter()
        return loaded_params

    @staticmethod
    def _validate_loaded_weights(model: nn.Module, loaded_params: set[str]) -> None:
        expected_params = set(model.state_dict())
        expected_params = {
            name
            for name in expected_params
            if not name.endswith("rotary_emb.inv_freq")
        }

        tie_word_embeddings = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
        if tie_word_embeddings:
            expected_params.discard("lm_head.weight")

        missing_params = sorted(expected_params - loaded_params)
        if missing_params:
            raise ValueError(
                "Missing weights after Hugging Face load: "
                + ", ".join(missing_params[:20])
                + (" ..." if len(missing_params) > 20 else "")
            )


class HuggingFaceModelLoader(DefaultModelLoader):
    """Compatibility alias for the default Hugging Face loader."""


def _resolve_model_dtype(hf_config: PretrainedConfig) -> torch.dtype | None:
    dtype = getattr(hf_config, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype, None)
    return None


@contextmanager
def _temporary_default_dtype(dtype: torch.dtype | None):
    if dtype is None or not dtype.is_floating_point:
        yield
        return

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)
