from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors_file
from transformers import LlamaConfig, MistralConfig, Qwen3Config

from kvcore.config import DeviceConfig, KVCoreConfig, LoadConfig, ModelConfig
from kvcore.model.model_loader import DefaultModelLoader
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


def _make_kvcore_config(config, *, attn_backend: str | None = None) -> KVCoreConfig:
    kvcore_config = KVCoreConfig(
        model_config=ModelConfig(
            model="unused",
            attn_backend=attn_backend or "torch_paged",
            hf_config=config,
        )
    )
    return kvcore_config


def _make_loader_config(
    model_dir: Path,
    *,
    load_format: str = "auto",
    device: str | None = None,
) -> KVCoreConfig:
    return KVCoreConfig(
        model_config=ModelConfig(model=str(model_dir)),
        load_config=LoadConfig(
            local_files_only=True,
            load_format=load_format,
        ),
        device_config=DeviceConfig(device=device),
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


def _write_sharded_safetensors_dir(root: Path, config, model) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(root)
    state_items = list(model.state_dict().items())
    split_at = len(state_items) // 2
    shards = {
        "model-00001-of-00002.safetensors": dict(state_items[:split_at]),
        "model-00002-of-00002.safetensors": dict(state_items[split_at:]),
    }
    weight_map: dict[str, str] = {}
    for shard_name, shard_state in shards.items():
        save_safetensors_file(shard_state, str(root / shard_name))
        weight_map.update({name: shard_name for name in shard_state})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return root


def test_hf_loader_creates_llama_model_from_local_config(tmp_path: Path) -> None:
    config = _make_tiny_llama_config()
    model_dir = tmp_path / "llama_local"
    config.save_pretrained(model_dir)

    loader = DefaultModelLoader(_make_loader_config(model_dir))

    hf_config = loader._load_config()
    model = loader._initialize_model(hf_config)

    assert isinstance(model, Llama3ForCausalLM)


def test_hf_loader_loads_pytorch_bin_weights(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference_model = Llama3ForCausalLM(_make_kvcore_config(_make_tiny_llama_config()))
    model_dir = _write_pretrained_dir(
        tmp_path / "llama_bin",
        reference_model.config,
        reference_model,
        use_safetensors=False,
    )

    loader = DefaultModelLoader(_make_loader_config(model_dir))
    loaded_model = loader.load_model()

    assert isinstance(loaded_model, Llama3ForCausalLM)
    _assert_same_state_dict(reference_model, loaded_model)


def test_hf_loader_loads_safetensors_weights(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference_model = Qwen3ForCausalLM(_make_kvcore_config(_make_tiny_qwen3_config()))
    model_dir = _write_pretrained_dir(
        tmp_path / "qwen3_safe",
        reference_model.config,
        reference_model,
        use_safetensors=True,
    )

    loader = DefaultModelLoader(_make_loader_config(model_dir))
    loaded_model = loader.load_model()

    assert isinstance(loaded_model, Qwen3ForCausalLM)
    _assert_same_state_dict(reference_model, loaded_model)


def test_default_loader_loads_sharded_safetensors_with_multithread(
    tmp_path: Path,
    monkeypatch,
) -> None:
    torch.manual_seed(0)
    reference_model = Qwen3ForCausalLM(_make_kvcore_config(_make_tiny_qwen3_config()))
    model_dir = _write_sharded_safetensors_dir(
        tmp_path / "qwen3_sharded_safe",
        reference_model.config,
        reference_model,
    )
    monkeypatch.setenv("KVCORE_WEIGHT_LOAD_THREADS", "2")

    loader = DefaultModelLoader(_make_loader_config(model_dir, load_format="safetensors"))
    loaded_model = loader.load_model()

    assert isinstance(loaded_model, Qwen3ForCausalLM)
    _assert_same_state_dict(reference_model, loaded_model)


def test_model_runner_load_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference_model = MistralForCausalLM(_make_kvcore_config(_make_tiny_mistral_config()))
    model_dir = _write_pretrained_dir(
        tmp_path / "mistral_bin",
        reference_model.config,
        reference_model,
        use_safetensors=False,
    )

    runner = ModelRunner(_make_loader_config(model_dir))

    loaded_model = runner.load_model()
    assert isinstance(loaded_model, MistralForCausalLM)
    _assert_same_state_dict(reference_model, loaded_model)


def test_llama_load_weights_stacks_hf_qkv_and_mlp_weights() -> None:
    config = _make_tiny_llama_config()
    model = Llama3ForCausalLM(_make_kvcore_config(config))
    layer = model.model.layers[0]
    q_weight = torch.full_like(layer.self_attn.qkv_proj.weight[:64], 1.0)
    k_weight = torch.full_like(layer.self_attn.qkv_proj.weight[64:96], 2.0)
    v_weight = torch.full_like(layer.self_attn.qkv_proj.weight[96:], 3.0)
    gate_weight = torch.full_like(layer.mlp.gate_up_proj.weight[:128], 4.0)
    up_weight = torch.full_like(layer.mlp.gate_up_proj.weight[128:], 5.0)

    model.load_weights(
        [
            ("model.layers.0.self_attn.q_proj.weight", q_weight),
            ("model.layers.0.self_attn.k_proj.weight", k_weight),
            ("model.layers.0.self_attn.v_proj.weight", v_weight),
            ("model.layers.0.mlp.gate_proj.weight", gate_weight),
            ("model.layers.0.mlp.up_proj.weight", up_weight),
        ]
    )

    assert torch.equal(layer.self_attn.qkv_proj.weight[:64], q_weight)
    assert torch.equal(layer.self_attn.qkv_proj.weight[64:96], k_weight)
    assert torch.equal(layer.self_attn.qkv_proj.weight[96:], v_weight)
    assert torch.equal(layer.mlp.gate_up_proj.weight[:128], gate_weight)
    assert torch.equal(layer.mlp.gate_up_proj.weight[128:], up_weight)


def test_default_loader_supports_pt_load_format(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference_model = Llama3ForCausalLM(_make_kvcore_config(_make_tiny_llama_config()))
    model_dir = _write_pretrained_dir(
        tmp_path / "llama_pt",
        reference_model.config,
        reference_model,
        use_safetensors=False,
    )

    loader = DefaultModelLoader(_make_loader_config(model_dir, load_format="hf"))
    loaded_model = loader.load_model()

    assert isinstance(loaded_model, Llama3ForCausalLM)
    _assert_same_state_dict(reference_model, loaded_model)
