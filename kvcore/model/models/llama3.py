from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig

from kvcore.config import KVCoreConfig
from kvcore.model.attn_backend import AttentionBackend, AttentionType
from kvcore.model.layer.activation import SiluAndMul
from kvcore.model.layer.attention import Attention
from kvcore.model.layer.linear import ColumnLinear, RowLinear
from kvcore.model.layer.rmsnorm import RMSNorm
from kvcore.model.layer.rotary_embedding import RotaryEmbedding
from kvcore.model.model_utils import (
    get_hf_config,
    load_named_weights,
    maybe_prefix,
    prepare_model_inputs,
)
from kvcore.sample import LogitsProcessor


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported for now."
            )

        mlp_bias = getattr(config, "mlp_bias", False)
        self.gate_up_proj = ColumnLinear(
            config.hidden_size,
            config.intermediate_size * 2,
            bias=mlp_bias,
        )
        self.down_proj = RowLinear(config.intermediate_size, config.hidden_size, bias=mlp_bias)
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.gate_up_proj(hidden_states))
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        prefix: str,
        attn_backend: str | AttentionBackend | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.scaling = self.head_dim**-0.5

        attention_bias = getattr(config, "attention_bias", False)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = ColumnLinear(
            self.hidden_size,
            self.q_size + (2 * self.kv_size),
            bias=attention_bias,
        )
        self.o_proj = RowLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
        )

        self.rotary_emb = RotaryEmbedding(config)
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            prefix=maybe_prefix(prefix, "attn"),
            attn_type=AttentionType.DECODER,
            attn_backend=attn_backend,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        query_states, key_states, value_states = self.qkv_proj(hidden_states).split(
            [self.q_size, self.kv_size, self.kv_size],
            dim=-1,
        )
        query_states, key_states = self.rotary_emb(positions, query_states, key_states)

        attn_output = self.attn(
            query_states,
            key_states,
            value_states,
        )
        output = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        prefix: str,
        attn_backend: str | AttentionBackend | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(
            config=config,
            prefix=maybe_prefix(prefix, "self_attn"),
            attn_backend=attn_backend,
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Llama3Model(nn.Module):
    """Minimal decoder-only Llama3 model aligned with HF/vLLM module layout."""

    def __init__(
        self,
        kvcore_config: KVCoreConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = get_hf_config(kvcore_config)
        if not isinstance(config, LlamaConfig):
            raise TypeError(f"Expected LlamaConfig, got {type(config)!r}")
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    config=config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx}"),
                    attn_backend=kvcore_config.model_config.attn_backend,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = prepare_model_inputs(self, input_ids, inputs_embeds)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Llama3ForCausalLM(nn.Module):
    def __init__(
        self,
        kvcore_config: KVCoreConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = get_hf_config(kvcore_config)
        if not isinstance(config, LlamaConfig):
            raise TypeError(f"Expected LlamaConfig, got {type(config)!r}")
        self.config = config
        self.model = Llama3Model(
            kvcore_config=kvcore_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(scale=logit_scale)

        # TODO 我对此有疑惑，不懂其中原理
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(input_ids)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_output = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        return model_output

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights = load_named_weights(
            self,
            weights,
            tied_lm_head=self.config.tie_word_embeddings,
            stacked_params_mapping=(
                (".qkv_proj", ".q_proj", "q"),
                (".qkv_proj", ".k_proj", "k"),
                (".qkv_proj", ".v_proj", "v"),
                (".gate_up_proj", ".gate_proj", 0),
                (".gate_up_proj", ".up_proj", 1),
            ),
        )
        return loaded_weights
