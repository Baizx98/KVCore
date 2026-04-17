"""Model-family registry and factory helpers."""

from __future__ import annotations

from typing import Any

from kvcore.api.config import EngineConfig
from kvcore.model.llama3 import Llama3Model
from kvcore.model.mistral3 import Mistral3Model
from kvcore.model.qwen3 import Qwen3Model

MODEL_FAMILIES = (
    Llama3Model,
    Qwen3Model,
    Mistral3Model,
)


def get_model_class(model_type: str):
    """Return the registered model wrapper class for one HF model type."""

    for model_cls in MODEL_FAMILIES:
        if model_cls.supports_model_type(model_type):
            return model_cls
    raise ValueError(f"unsupported Hugging Face model_type={model_type!r}")


def load_model_from_config(config: EngineConfig) -> Any:
    """Load one supported model family from Hugging Face metadata."""

    try:
        from transformers import AutoConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "transformers must be installed before loading a model"
        ) from exc

    hf_config = AutoConfig.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    model_cls = get_model_class(str(hf_config.model_type))
    return model_cls.from_config(config, hf_config=hf_config)
