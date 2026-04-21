from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors_file
from transformers import LlamaConfig, MistralConfig, Qwen3Config

from kvcore.model.model_loader import (
    DefaultModelLoader,
    HuggingFaceModelLoader,
    ModelLoadConfig,
)
from kvcore.model.model_runner import ModelRunner
from kvcore.model.models.llama3 import Llama3ForCausalLM
from kvcore.model.models.mistral import MistralForCausalLM
from kvcore.model.models.qwen3 import Qwen3ForCausalLM


def _make_tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )


def _make_tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        use_sliding_window=False,
    )


def _make_tiny_mistral_config() -> MistralConfig:
    return MistralConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        sliding_window=None,
    )


def _assert_same_state_dict(model_a, model_b) -> None:
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    assert state_dict_a.keys() == state_dict_b.keys()
    for name in state_dict_a:
        assert torch.equal(state_dict_a[name], state_dict_b[name]), name


def _write_pretrained_dir(
    root: Path,
    config,
    model,
    *,
    use_safetensors: bool,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(root)
    state_dict = model.state_dict()
    if use_safetensors:
        save_safetensors_file(state_dict, str(root / "model.safetensors"))
    else:
        torch.save(state_dict, root / "pytorch_model.bin")
    return root


def test_hf_loader_creates_llama_model_from_local_config(tmp_path: Path) -> None:
    config = _make_tiny_llama_config()
    model_dir = tmp_path / "llama_local"
    config.save_pretrained(model_dir)

    loader = HuggingFaceModelLoader(
        ModelLoadConfig(
            model=str(model_dir),
            local_files_only=True,
        )
    )

    hf_config = loader.load_config_from_source()
    model = loader.create_model(hf_config)

    assert isinstance(model, Llama3ForCausalLM)


def test_hf_loader_loads_pytorch_bin_weights(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference_model = Llama3ForCausalLM(_make_tiny_llama_config())
    model_dir = _write_pretrained_dir(
        tmp_path / "llama_bin",
        reference_model.config,
        reference_model,
        use_safetensors=False,
    )

    loader = HuggingFaceModelLoader(
        ModelLoadConfig(
            model=str(model_dir),
            local_files_only=True,
        )
    )
    loaded_model = loader.load_model()

    assert isinstance(loaded_model, Llama3ForCausalLM)
    _assert_same_state_dict(reference_model, loaded_model)


def test_hf_loader_loads_safetensors_weights(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference_model = Qwen3ForCausalLM(_make_tiny_qwen3_config())
    model_dir = _write_pretrained_dir(
        tmp_path / "qwen3_safe",
        reference_model.config,
        reference_model,
        use_safetensors=True,
    )

    loader = HuggingFaceModelLoader(
        ModelLoadConfig(
            model=str(model_dir),
            local_files_only=True,
        )
    )
    loaded_model = loader.load_model()

    assert isinstance(loaded_model, Qwen3ForCausalLM)
    _assert_same_state_dict(reference_model, loaded_model)


def test_model_runner_create_model_then_load_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference_model = MistralForCausalLM(_make_tiny_mistral_config())
    model_dir = _write_pretrained_dir(
        tmp_path / "mistral_bin",
        reference_model.config,
        reference_model,
        use_safetensors=False,
    )

    runner = ModelRunner(
        ModelLoadConfig(
            model=str(model_dir),
            local_files_only=True,
        )
    )

    empty_model = runner.create_model()
    assert isinstance(empty_model, MistralForCausalLM)

    loaded_model = runner.load_model()
    assert isinstance(loaded_model, MistralForCausalLM)
    _assert_same_state_dict(reference_model, loaded_model)


def test_default_loader_supports_pt_load_format(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference_model = Llama3ForCausalLM(_make_tiny_llama_config())
    model_dir = _write_pretrained_dir(
        tmp_path / "llama_pt",
        reference_model.config,
        reference_model,
        use_safetensors=False,
    )

    loader = DefaultModelLoader(
        ModelLoadConfig(
            model=str(model_dir),
            local_files_only=True,
            load_format="hf",
        )
    )
    loaded_model = loader.load_model()

    assert isinstance(loaded_model, Llama3ForCausalLM)
    _assert_same_state_dict(reference_model, loaded_model)
