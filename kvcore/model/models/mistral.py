from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from transformers.models.mistral.configuration_mistral import MistralConfig

from kvcore.model.attn_backend import AttentionBackend, AttentionType
from kvcore.model.layer.activation import SiluAndMul
from kvcore.model.layer.attention import Attention
from kvcore.model.layer.linear import ColumnLinear, RowLinear
from kvcore.model.layer.rmsnorm import RMSNorm
from kvcore.model.layer.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb
from kvcore.model.model_utils import (
    apply_sliding_window_metadata,
    infer_batch_and_seq_len,
    load_named_weights,
    prepare_model_inputs,
)


class MistralMLP(nn.Module):
    def __init__(self, config: MistralConfig) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported for now."
            )

        self.gate_proj = ColumnLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ColumnLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = RowLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.gate_proj(hidden_states), self.up_proj(hidden_states))
        return self.down_proj(hidden_states)


class MistralAttention(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        layer_idx: int,
        attn_backend: str | AttentionBackend | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.scaling = self.head_dim**-0.5
        self.sliding_window = getattr(config, "sliding_window", None)

        self.q_proj = ColumnLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = ColumnLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = ColumnLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = RowLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(config)
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            prefix=f"layers.{layer_idx}.self_attn",
            attn_type=AttentionType.DECODER,
            attn_backend=attn_backend,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = infer_batch_and_seq_len(hidden_states)
        input_shape = hidden_states.shape[:-1]

        query_states = self.q_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(
            positions,
            batch_size=batch_size,
            seq_len=seq_len,
            device=query_states.device,
            dtype=query_states.dtype,
        )
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = self.attn(
            query_states,
            key_states,
            value_states,
            output_shape=torch.Size((*input_shape, self.num_heads * self.head_dim)),
            attn_metadata=apply_sliding_window_metadata(attn_metadata, self.sliding_window),
            layer_idx=self.layer_idx,
        )
        return self.o_proj(attn_output)


class MistralDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        layer_idx: int,
        attn_backend: str | AttentionBackend | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = MistralAttention(
            config=config,
            layer_idx=layer_idx,
            attn_backend=attn_backend,
        )
        self.mlp = MistralMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MistralModel(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        *,
        attn_backend: str | AttentionBackend | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [
                MistralDecoderLayer(config=config, layer_idx=layer_idx, attn_backend=attn_backend)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        hidden_states = prepare_model_inputs(self, input_ids, inputs_embeds)

        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
            )

        return self.norm(hidden_states)


class MistralForCausalLM(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        *,
        attn_backend: str | AttentionBackend | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = MistralModel(config=config, attn_backend=attn_backend)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            attn_metadata=attn_metadata,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_named_weights(
            self,
            weights,
            tied_lm_head=self.config.tie_word_embeddings,
        )
