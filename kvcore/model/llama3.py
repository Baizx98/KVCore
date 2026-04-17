"""vLLM-style Llama3 model implementation with manual layers."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from transformers import AutoModelForCausalLM, LlamaConfig

from kvcore.model.base import HFDecoderModel
from kvcore.model.layers import (
    DecoderLayer,
    DecoderMLP,
    ParallelLMHead,
    RMSNorm,
    VocabEmbedding,
)
from kvcore.model.metadata import AttentionMetadata, ModelKVCacheSpec


class LlamaMLP(nn.Module):
    """Llama MLP wrapper using the shared decoder MLP implementation."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.inner = DecoderMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=bool(getattr(config, "mlp_bias", False)),
        )

    @property
    def gate_up_proj(self):
        return self.inner.gate_up_proj

    @property
    def down_proj(self):
        return self.inner.down_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.inner(hidden_states)


class LlamaAttention(nn.Module):
    """Llama attention with vLLM-like input signature."""

    def __init__(self, config: LlamaConfig, *, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        rope_theta = float(getattr(config, "rope_theta", 10000.0))
        attention_bias = bool(getattr(config, "attention_bias", False))
        self.inner = DecoderLayer(
            hidden_size=hidden_size,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=attention_bias,
            mlp_bias=bool(getattr(config, "mlp_bias", False)),
            layer_idx=layer_idx,
        ).self_attn
        self.qkv_proj = self.inner.qkv_proj
        self.o_proj = self.inner.o_proj
        self.rotary_emb = self.inner.rotary_emb
        self.q_size = self.inner.q_size
        self.kv_size = self.inner.kv_size

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_caches,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return self.inner(positions, hidden_states, kv_caches, attn_metadata)


class LlamaDecoderLayer(nn.Module):
    """Llama decoder layer following vLLM's module structure."""

    def __init__(self, config: LlamaConfig, *, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        kv_caches,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, kv_caches, attn_metadata)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    """Flattened-token Llama model body."""

    def __init__(self, hf_config: LlamaConfig) -> None:
        super().__init__()
        self.config = hf_config
        self.embed_tokens = VocabEmbedding(hf_config.vocab_size, hf_config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(hf_config, layer_idx=idx)
            for idx in range(hf_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        kv_caches,
        attn_metadata: AttentionMetadata,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            hidden_states = self.embed_input_ids(input_ids)
        else:
            hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                kv_caches,
                attn_metadata,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    """Top-level Llama causal LM."""

    def __init__(self, hf_config: LlamaConfig) -> None:
        super().__init__()
        self.config = hf_config
        self.model = LlamaModel(hf_config)
        self.lm_head = ParallelLMHead(hf_config.hidden_size, hf_config.vocab_size, bias=False)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        kv_caches,
        attn_metadata: AttentionMetadata,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, kv_caches, attn_metadata, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def get_kv_cache_spec(self, *, block_size: int, dtype: torch.dtype) -> ModelKVCacheSpec:
        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        return ModelKVCacheSpec(
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=getattr(
                self.config,
                "num_key_value_heads",
                self.config.num_attention_heads,
            ),
            head_dim=head_dim,
            block_size=block_size,
            cache_dtype=str(dtype).replace("torch.", ""),
            attention_dtype=str(dtype).replace("torch.", ""),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ".q_proj." in name:
                _load_packed_param(params, name, loaded_weight, ".q_proj.", ".qkv_proj.", "q")
            elif ".k_proj." in name:
                _load_packed_param(params, name, loaded_weight, ".k_proj.", ".qkv_proj.", "k")
            elif ".v_proj." in name:
                _load_packed_param(params, name, loaded_weight, ".v_proj.", ".qkv_proj.", "v")
            elif ".gate_proj." in name:
                _load_packed_param(params, name, loaded_weight, ".gate_proj.", ".gate_up_proj.", 0)
            elif ".up_proj." in name:
                _load_packed_param(params, name, loaded_weight, ".up_proj.", ".gate_up_proj.", 1)
            else:
                target = params.get(name)
                if target is not None:
                    target.data.copy_(loaded_weight.to(dtype=target.dtype))

    def load_weights_from_hf(
        self,
        *,
        model_name_or_path: str,
        trust_remote_code: bool,
        dtype: torch.dtype,
    ) -> None:
        reference_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        self.load_weights(reference_model.state_dict().items())


class Llama3Model(HFDecoderModel):
    """Tokenizer + top-level Llama3 model bundle."""

    family_name = "llama3"
    supported_model_types = ("llama",)
    model_cls = LlamaForCausalLM


def _load_packed_param(
    params: dict[str, nn.Parameter],
    name: str,
    loaded_weight: torch.Tensor,
    source_name: str,
    packed_name: str,
    shard_id: str | int,
) -> None:
    target_name = name.replace(source_name, packed_name)
    target = params.get(target_name)
    if target is None:
        return
    if shard_id == "q":
        start = 0
        end = loaded_weight.shape[0]
    elif shard_id == "k":
        start = target.shape[0] // 3
        end = start + loaded_weight.shape[0]
    elif shard_id == "v":
        start = target.shape[0] - loaded_weight.shape[0]
        end = target.shape[0]
    elif shard_id == 0:
        start = 0
        end = loaded_weight.shape[0]
    else:
        start = target.shape[0] // 2
        end = target.shape[0]
    target.data[start:end].copy_(loaded_weight.to(dtype=target.dtype))
