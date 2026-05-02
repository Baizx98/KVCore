from __future__ import annotations

import torch
from transformers import LlamaConfig, MistralConfig, Qwen3Config

from kvcore.config import KVCoreConfig, ModelConfig
from kvcore.model.forward_context import ForwardContext, set_forward_context
from kvcore.model.layer.attention import Attention
from kvcore.model.models.llama3 import Llama3ForCausalLM
from kvcore.model.models.mistral import MistralForCausalLM
from kvcore.model.models.qwen3 import Qwen3ForCausalLM


def make_tiny_llama3_config(**overrides) -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_tiny_qwen3_config(**overrides) -> Qwen3Config:
    config = Qwen3Config(
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
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_tiny_mistral_config(**overrides) -> MistralConfig:
    config = MistralConfig(
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
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_kvcore_config(hf_config, *, attn_backend=None) -> KVCoreConfig:
    kvcore_config = KVCoreConfig(
        model_config=ModelConfig(
            model="unused",
            attn_backend=attn_backend or "torch_paged",
            hf_config=hf_config,
        )
    )
    return kvcore_config


class ZeroAttentionBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        num_kv_heads: int,
        scaling: float,
        is_causal: bool,
        attn_metadata: object | None = None,
        layer_idx: int | None = None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value, num_kv_heads, scaling, is_causal
        self.calls.append(
            {
                "query_shape": tuple(query.shape),
                "attn_metadata": attn_metadata,
                "layer_idx": layer_idx,
            }
        )
        if output is None:
            output = torch.zeros_like(query)
        else:
            output.zero_()
        return output


def test_attention_wrapper_flat_token_flow() -> None:
    torch.manual_seed(0)
    layer = Attention(
        num_heads=4,
        head_size=16,
        scale=16**-0.5,
        num_kv_heads=2,
        attn_backend=ZeroAttentionBackend(),
        prefix="model.layers.3.self_attn.attn",
    )
    attn_metadata = {"tag": "attention-wrapper"}

    query = torch.randn(5, 64)
    key = torch.randn(5, 32)
    value = torch.randn(5, 32)

    with set_forward_context(ForwardContext(attn_metadata=attn_metadata)):
        output = layer(
            query,
            key,
            value,
        )

    assert output.shape == (5, 64)
    backend = layer.get_attn_backend()
    assert isinstance(backend, ZeroAttentionBackend)
    assert len(backend.calls) == 1
    assert backend.calls[0]["layer_idx"] == 3
    assert backend.calls[0]["attn_metadata"] is attn_metadata
    assert backend.calls[0]["query_shape"] == (5, 4, 16)


def test_attention_wrapper_requires_forward_context() -> None:
    layer = Attention(
        num_heads=4,
        head_size=16,
        scale=16**-0.5,
        num_kv_heads=2,
        attn_backend=ZeroAttentionBackend(),
        prefix="model.layers.0.self_attn.attn",
    )
    query = torch.randn(1, 64)
    key = torch.randn(1, 32)
    value = torch.randn(1, 32)

    try:
        layer(query, key, value)
    except RuntimeError as exc:
        assert "Forward context is not set" in str(exc)
    else:
        raise AssertionError("Attention should require a forward context")


def test_llama3_flat_forward_and_logits_shape() -> None:
    torch.manual_seed(0)
    model = Llama3ForCausalLM(
        make_kvcore_config(make_tiny_llama3_config(), attn_backend=ZeroAttentionBackend())
    )
    model.eval()

    input_ids = torch.randint(0, model.config.vocab_size, (5,))
    positions = torch.arange(input_ids.size(0))

    with set_forward_context(ForwardContext(attn_metadata={"tag": "llama3-flat"})):
        hidden_states = model(input_ids=input_ids, positions=positions)
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (5, model.config.hidden_size)
    assert logits.shape == (5, model.config.vocab_size)


def test_llama3_tie_word_embeddings() -> None:
    model = Llama3ForCausalLM(make_kvcore_config(make_tiny_llama3_config(tie_word_embeddings=True)))

    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_llama3_attention_layer_idx_is_derived_from_prefix() -> None:
    model = Llama3ForCausalLM(
        make_kvcore_config(make_tiny_llama3_config(), attn_backend=ZeroAttentionBackend())
    )

    assert model.model.layers[0].self_attn.attn.layer_idx == 0
    assert model.model.layers[1].self_attn.attn.layer_idx == 1
    assert model.model.layers[1].self_attn.attn.layer_name == "model.layers.1.self_attn.attn"


def test_qwen3_flat_forward_and_logits_shape() -> None:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(
        make_kvcore_config(make_tiny_qwen3_config(), attn_backend=ZeroAttentionBackend())
    )
    model.eval()

    input_ids = torch.randint(0, model.config.vocab_size, (5,))
    positions = torch.arange(input_ids.size(0))

    with set_forward_context(ForwardContext(attn_metadata={"tag": "qwen3"})):
        hidden_states = model(input_ids=input_ids, positions=positions)
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (5, model.config.hidden_size)
    assert logits.shape == (5, model.config.vocab_size)


def test_mistral_flat_forward_and_logits_shape() -> None:
    torch.manual_seed(0)
    model = MistralForCausalLM(
        make_kvcore_config(make_tiny_mistral_config(), attn_backend=ZeroAttentionBackend())
    )
    model.eval()

    input_ids = torch.randint(0, model.config.vocab_size, (5,))
    positions = torch.arange(input_ids.size(0))

    with set_forward_context(ForwardContext(attn_metadata={"tag": "mistral"})):
        hidden_states = model(input_ids=input_ids, positions=positions)
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (5, model.config.hidden_size)
    assert logits.shape == (5, model.config.vocab_size)
