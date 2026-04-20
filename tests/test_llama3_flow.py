from __future__ import annotations

import torch
from transformers import LlamaConfig, MistralConfig, Qwen3Config

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


class RecordingKVCache:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: object | None = None,
        layer_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(
            {
                "key_shape": tuple(key.shape),
                "value_shape": tuple(value.shape),
                "attn_metadata": attn_metadata,
                "layer_idx": layer_idx,
            }
        )
        return key, value


def test_attention_wrapper_batched_flow() -> None:
    torch.manual_seed(0)
    layer = Attention(
        num_heads=4,
        head_size=16,
        scale=16**-0.5,
        num_kv_heads=2,
    )
    kv_cache = RecordingKVCache()
    attn_metadata = {"tag": "attention-wrapper"}

    query = torch.randn(2, 4, 5, 16)
    key = torch.randn(2, 2, 5, 16)
    value = torch.randn(2, 2, 5, 16)

    output = layer(
        query,
        key,
        value,
        output_shape=torch.Size((2, 5, 64)),
        attn_metadata=attn_metadata,
        kv_cache=kv_cache,
        layer_idx=3,
    )

    assert output.shape == (2, 5, 64)
    assert len(kv_cache.calls) == 1
    assert kv_cache.calls[0]["layer_idx"] == 3
    assert kv_cache.calls[0]["attn_metadata"] is attn_metadata


def test_attention_wrapper_uses_bound_kv_cache() -> None:
    torch.manual_seed(0)
    layer = Attention(
        num_heads=4,
        head_size=16,
        scale=16**-0.5,
        num_kv_heads=2,
    )
    kv_cache = RecordingKVCache()
    layer.bind_kv_cache(kv_cache)

    query = torch.randn(2, 4, 3, 16)
    key = torch.randn(2, 2, 3, 16)
    value = torch.randn(2, 2, 3, 16)

    output = layer(
        query,
        key,
        value,
        output_shape=torch.Size((2, 3, 64)),
        layer_idx=1,
    )

    assert output.shape == (2, 3, 64)
    assert len(kv_cache.calls) == 1
    assert kv_cache.calls[0]["layer_idx"] == 1


def test_llama3_batched_forward_and_logits_shape() -> None:
    torch.manual_seed(0)
    model = Llama3ForCausalLM(make_tiny_llama3_config())
    model.eval()

    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    positions = torch.arange(input_ids.size(1)).unsqueeze(0).expand(input_ids.size(0), -1)

    hidden_states = model(input_ids=input_ids, positions=positions)
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (2, 5, model.config.hidden_size)
    assert logits.shape == (2, 5, model.config.vocab_size)


def test_llama3_flat_forward_and_logits_shape() -> None:
    torch.manual_seed(0)
    model = Llama3ForCausalLM(make_tiny_llama3_config())
    model.eval()

    input_ids = torch.randint(0, model.config.vocab_size, (5,))
    positions = torch.arange(input_ids.size(0))

    hidden_states = model(input_ids=input_ids, positions=positions)
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (5, model.config.hidden_size)
    assert logits.shape == (5, model.config.vocab_size)


def test_llama3_tie_word_embeddings() -> None:
    model = Llama3ForCausalLM(make_tiny_llama3_config(tie_word_embeddings=True))

    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_llama3_propagates_kv_cache_updates() -> None:
    torch.manual_seed(0)
    model = Llama3ForCausalLM(make_tiny_llama3_config())
    model.eval()

    input_ids = torch.randint(0, model.config.vocab_size, (2, 4))
    positions = torch.arange(input_ids.size(1)).unsqueeze(0).expand(input_ids.size(0), -1)
    attn_metadata = {"tag": "flow-test"}
    kv_caches = [RecordingKVCache() for _ in range(model.config.num_hidden_layers)]

    hidden_states = model(
        input_ids=input_ids,
        positions=positions,
        attn_metadata=attn_metadata,
        kv_caches=kv_caches,
    )

    assert hidden_states.shape == (2, 4, model.config.hidden_size)
    for layer_idx, cache in enumerate(kv_caches):
        assert len(cache.calls) == 1
        assert cache.calls[0]["layer_idx"] == layer_idx
        assert cache.calls[0]["attn_metadata"] is attn_metadata
        assert cache.calls[0]["key_shape"] == (2, model.config.num_key_value_heads, 4, 16)
        assert cache.calls[0]["value_shape"] == (2, model.config.num_key_value_heads, 4, 16)


def test_qwen3_batched_forward_and_logits_shape() -> None:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(make_tiny_qwen3_config())
    model.eval()

    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    positions = torch.arange(input_ids.size(1)).unsqueeze(0).expand(input_ids.size(0), -1)

    hidden_states = model(input_ids=input_ids, positions=positions)
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (2, 5, model.config.hidden_size)
    assert logits.shape == (2, 5, model.config.vocab_size)


def test_mistral_batched_forward_and_logits_shape() -> None:
    torch.manual_seed(0)
    model = MistralForCausalLM(make_tiny_mistral_config())
    model.eval()

    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    positions = torch.arange(input_ids.size(1)).unsqueeze(0).expand(input_ids.size(0), -1)

    hidden_states = model(input_ids=input_ids, positions=positions)
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (2, 5, model.config.hidden_size)
    assert logits.shape == (2, 5, model.config.vocab_size)
