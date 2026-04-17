from __future__ import annotations

from typing import Any

import torch
from transformers import LlamaConfig, Qwen2Config

from kvcore.model import (
    AttentionMetadata,
    LlamaAttention,
    LlamaForCausalLM,
    Qwen3ForCausalLM,
)
from kvcore.model.layers import BlockedKVCache, BlockedKVCacheCollection


def _make_llama_config() -> Any:
    config = LlamaConfig()
    config.hidden_size = 8
    config.intermediate_size = 16
    config.num_hidden_layers = 1
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.vocab_size = 32
    config.max_position_embeddings = 32
    return config


def _make_llama_lm_config() -> Any:
    config = LlamaConfig()
    config.hidden_size = 8
    config.intermediate_size = 16
    config.num_hidden_layers = 2
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.vocab_size = 32
    config.max_position_embeddings = 32
    return config


def _make_qwen_config() -> Any:
    config = Qwen2Config()
    config.hidden_size = 8
    config.intermediate_size = 16
    config.num_hidden_layers = 2
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.vocab_size = 32
    config.max_position_embeddings = 32
    return config


def test_llama_attention_supports_flattened_prefill_batch_and_blocked_kv_cache() -> None:
    config = _make_llama_config()
    attention = LlamaAttention(config, layer_idx=0)
    kv_caches = BlockedKVCacheCollection(
        layers=[
            BlockedKVCache.allocate(
                num_blocks=4,
                block_size=2,
                num_kv_heads=1,
                head_dim=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
        ]
    )
    hidden_states = torch.randn(3, 8)
    positions = torch.tensor([0, 1, 0], dtype=torch.long)
    attn_metadata = AttentionMetadata(
        query_start_locs=torch.tensor([0, 2, 3], dtype=torch.long),
        seq_lens=torch.tensor([2, 1], dtype=torch.long),
        context_lens=torch.tensor([0, 0], dtype=torch.long),
        block_tables=torch.tensor([[0, -1], [1, -1]], dtype=torch.long),
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.long),
        block_size=2,
        max_query_len=2,
        max_seq_len=2,
        num_prefill_sequences=2,
        num_decode_tokens=0,
    )

    output = attention(positions, hidden_states, kv_caches, attn_metadata)

    assert output.shape == (3, 8)
    assert kv_caches.layers[0].key_cache[0, 0].abs().sum() > 0
    assert kv_caches.layers[0].key_cache[0, 1].abs().sum() > 0
    assert kv_caches.layers[0].key_cache[1, 0].abs().sum() > 0


def test_llama_attention_supports_chunked_prefill_inputs() -> None:
    config = _make_llama_config()
    attention = LlamaAttention(config, layer_idx=0)
    kv_caches = BlockedKVCacheCollection(
        layers=[
            BlockedKVCache.allocate(
                num_blocks=4,
                block_size=2,
                num_kv_heads=1,
                head_dim=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
        ],
        current_seq_len=2,
    )

    warmup_hidden = torch.randn(2, 8)
    warmup_positions = torch.tensor([0, 1], dtype=torch.long)
    warmup_metadata = AttentionMetadata(
        query_start_locs=torch.tensor([0, 2], dtype=torch.long),
        seq_lens=torch.tensor([2], dtype=torch.long),
        context_lens=torch.tensor([0], dtype=torch.long),
        block_tables=torch.tensor([[0, -1]], dtype=torch.long),
        slot_mapping=torch.tensor([0, 1], dtype=torch.long),
        block_size=2,
        max_query_len=2,
        max_seq_len=2,
        num_prefill_sequences=1,
        num_decode_tokens=0,
    )
    attention(warmup_positions, warmup_hidden, kv_caches, warmup_metadata)

    chunk_hidden = torch.randn(2, 8)
    chunk_positions = torch.tensor([2, 3], dtype=torch.long)
    chunk_metadata = AttentionMetadata(
        query_start_locs=torch.tensor([0, 2], dtype=torch.long),
        seq_lens=torch.tensor([4], dtype=torch.long),
        context_lens=torch.tensor([2], dtype=torch.long),
        block_tables=torch.tensor([[0, 1]], dtype=torch.long),
        slot_mapping=torch.tensor([2, 3], dtype=torch.long),
        block_size=2,
        max_query_len=2,
        max_seq_len=4,
        num_prefill_sequences=1,
        num_decode_tokens=0,
    )

    output = attention(chunk_positions, chunk_hidden, kv_caches, chunk_metadata)

    assert chunk_metadata.is_chunked_prefill
    assert output.shape == (2, 8)
    assert kv_caches.layers[0].key_cache[1, 0].abs().sum() > 0
    assert kv_caches.layers[0].key_cache[1, 1].abs().sum() > 0


def test_llama_for_causal_lm_accepts_flattened_inputs() -> None:
    config = _make_llama_lm_config()
    model = LlamaForCausalLM(config)
    kv_caches = BlockedKVCacheCollection(
        layers=[
            BlockedKVCache.allocate(
                num_blocks=4,
                block_size=2,
                num_kv_heads=1,
                head_dim=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            for _ in range(config.num_hidden_layers)
        ]
    )
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    positions = torch.tensor([0, 1, 0], dtype=torch.long)
    attn_metadata = AttentionMetadata(
        query_start_locs=torch.tensor([0, 2, 3], dtype=torch.long),
        seq_lens=torch.tensor([2, 1], dtype=torch.long),
        context_lens=torch.tensor([0, 0], dtype=torch.long),
        block_tables=torch.tensor([[0, -1], [1, -1]], dtype=torch.long),
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.long),
        block_size=2,
        max_query_len=2,
        max_seq_len=2,
        num_prefill_sequences=2,
        num_decode_tokens=0,
    )

    hidden_states = model(input_ids, positions, kv_caches, attn_metadata)
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (3, 8)
    assert logits.shape == (3, 32)


def test_qwen3_for_causal_lm_accepts_flattened_inputs() -> None:
    config = _make_qwen_config()
    model = Qwen3ForCausalLM(config)
    kv_caches = BlockedKVCacheCollection(
        layers=[
            BlockedKVCache.allocate(
                num_blocks=4,
                block_size=2,
                num_kv_heads=1,
                head_dim=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            for _ in range(config.num_hidden_layers)
        ]
    )
    input_ids = torch.tensor([4, 5], dtype=torch.long)
    positions = torch.tensor([0, 1], dtype=torch.long)
    attn_metadata = AttentionMetadata(
        query_start_locs=torch.tensor([0, 2], dtype=torch.long),
        seq_lens=torch.tensor([2], dtype=torch.long),
        context_lens=torch.tensor([0], dtype=torch.long),
        block_tables=torch.tensor([[0, -1]], dtype=torch.long),
        slot_mapping=torch.tensor([0, 1], dtype=torch.long),
        block_size=2,
        max_query_len=2,
        max_seq_len=2,
        num_prefill_sequences=1,
        num_decode_tokens=0,
    )

    hidden_states = model(input_ids, positions, kv_caches, attn_metadata)
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (2, 8)
    assert logits.shape == (2, 32)
