"""Base loader and compatibility wrapper for manual decoder-only models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from kvcore.api.config import EngineConfig
from kvcore.logging import get_logger
from kvcore.model.layers import (
    BlockedKVCache,
    BlockedKVCacheCollection,
    clear_attention_hook_state,
    install_attention_hook_state,
)
from kvcore.model.metadata import AttentionMetadata, ModelKVCacheSpec


@dataclass(slots=True)
class PreparedModelInputs:
    """Compatibility container used by the current model runner."""

    hidden_states: torch.Tensor
    causal_mask: torch.Tensor | None
    position_ids: torch.Tensor
    position_embeddings: None
    past_key_values: BlockedKVCacheCollection
    past_seq_len: int
    q_len: int
    attn_metadata: AttentionMetadata


@dataclass(slots=True)
class HFDecoderModel:
    """Tokenizer + manual model bundle loaded from Hugging Face checkpoints."""

    config: EngineConfig
    tokenizer: Any
    model: Any
    hf_config: Any
    device: str
    torch_dtype: torch.dtype
    num_hidden_layers: int
    model_type: str
    kv_cache_spec: ModelKVCacheSpec

    family_name: ClassVar[str] = "decoder"
    supported_model_types: ClassVar[tuple[str, ...]] = ()
    model_cls: ClassVar[type[Any]]

    @classmethod
    def supports_model_type(cls, model_type: str) -> bool:
        return model_type in cls.supported_model_types

    @classmethod
    def from_config(
        cls,
        config: EngineConfig,
        *,
        hf_config: Any | None = None,
    ) -> HFDecoderModel:
        try:
            from transformers import AutoConfig, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers must be installed before loading a model"
            ) from exc

        logger = get_logger(f"model.{cls.family_name}")
        logger.info("loading tokenizer and weights from %s", config.model_name_or_path)
        resolved_hf_config = hf_config or AutoConfig.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        if not cls.supports_model_type(str(resolved_hf_config.model_type)):
            raise ValueError(
                f"{cls.__name__} does not support model_type={resolved_hf_config.model_type!r}"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = _resolve_device(config.device)
        torch_dtype = _resolve_dtype(config.dtype)
        model = cls.model_cls(hf_config=resolved_hf_config)
        model.to(dtype=torch_dtype)
        model.load_weights_from_hf(
            model_name_or_path=config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            dtype=torch_dtype,
        )
        model.eval()
        model.to(device)

        kv_cache_spec = model.get_kv_cache_spec(block_size=config.block_size, dtype=torch_dtype)
        return cls(
            config=config,
            tokenizer=tokenizer,
            model=model,
            hf_config=resolved_hf_config,
            device=str(device),
            torch_dtype=torch_dtype,
            num_hidden_layers=kv_cache_spec.num_hidden_layers,
            model_type=str(resolved_hf_config.model_type),
            kv_cache_spec=kv_cache_spec,
        )

    def encode_prompt(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(text, return_tensors="pt")
        return {name: tensor.to(self.device) for name, tensor in encoded.items()}

    def decode_tokens(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def make_input_ids(self, token_ids: list[int], *, device: str) -> torch.Tensor:
        return torch.tensor([token_ids], device=device)

    def init_cache(self) -> BlockedKVCacheCollection:
        layers = [
            BlockedKVCache.allocate(
                num_blocks=128,
                block_size=self.kv_cache_spec.block_size,
                num_kv_heads=self.kv_cache_spec.num_key_value_heads,
                head_dim=self.kv_cache_spec.head_dim,
                dtype=self.torch_dtype,
                device=torch.device(self.device),
            )
            for _ in range(self.num_hidden_layers)
        ]
        return BlockedKVCacheCollection(layers=layers)

    def prepare_decoder_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: BlockedKVCacheCollection | None,
    ) -> PreparedModelInputs:
        del attention_mask
        cache = past_key_values if past_key_values is not None else self.init_cache()
        flat_input_ids = input_ids.reshape(-1)
        hidden_states = self.model.embed_input_ids(flat_input_ids)
        past_seq_len = cache.get_seq_length()
        q_len = int(flat_input_ids.shape[0])
        position_ids = torch.arange(
            past_seq_len,
            past_seq_len + q_len,
            device=hidden_states.device,
            dtype=torch.long,
        )
        attn_metadata = build_single_request_attn_metadata(
            positions=position_ids,
            seq_block_size=self.kv_cache_spec.block_size,
            past_seq_len=past_seq_len,
        )
        return PreparedModelInputs(
            hidden_states=hidden_states,
            causal_mask=None,
            position_ids=position_ids,
            position_embeddings=None,
            past_key_values=cache,
            past_seq_len=past_seq_len,
            q_len=q_len,
            attn_metadata=attn_metadata,
        )

    def iter_layers(self) -> tuple[Any, ...]:
        return tuple(self.model.model.layers)

    def run_layer(
        self,
        *,
        layer_module: Any,
        hidden_states: torch.Tensor,
        attention_mask: Any,
        position_ids: torch.Tensor,
        position_embeddings: Any,
        past_key_values: BlockedKVCacheCollection,
        attention_params: Any | None = None,
    ) -> torch.Tensor:
        del attention_mask, position_embeddings
        attn_metadata = (
            attention_params.attn_metadata
            if attention_params is not None and hasattr(attention_params, "attn_metadata")
            else build_single_request_attn_metadata(
                positions=position_ids,
                seq_block_size=self.kv_cache_spec.block_size,
                past_seq_len=past_key_values.get_seq_length(),
            )
        )
        hidden_states, residual = layer_module(
            positions=position_ids,
            hidden_states=hidden_states,
            residual=None,
            kv_caches=past_key_values,
            attn_metadata=attn_metadata,
        )
        return hidden_states + residual

    def before_attention(self, *, layer_module: Any, layer_context: Any) -> None:
        attention_module = getattr(layer_module, "self_attn", None)
        if attention_module is None:
            return
        install_attention_hook_state(
            attention_module=attention_module,
            attention_params=layer_context.attention_params,
        )

    def after_attention(self, *, layer_module: Any, layer_context: Any) -> None:
        attention_module = getattr(layer_module, "self_attn", None)
        if attention_module is None:
            return
        clear_attention_hook_state(attention_module=attention_module)

    def finalize_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.model.norm(hidden_states)

    def project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)


def build_single_request_attn_metadata(
    *,
    positions: torch.Tensor,
    seq_block_size: int,
    past_seq_len: int,
) -> AttentionMetadata:
    query_len = int(positions.shape[0])
    seq_len = past_seq_len + query_len
    num_blocks = (seq_len + seq_block_size - 1) // seq_block_size
    block_table = torch.arange(num_blocks, device=positions.device, dtype=torch.long).unsqueeze(0)
    slot_mapping = torch.arange(
        past_seq_len,
        past_seq_len + query_len,
        device=positions.device,
        dtype=torch.long,
    )
    return AttentionMetadata(
        query_start_locs=torch.tensor([0, query_len], device=positions.device, dtype=torch.long),
        seq_lens=torch.tensor([seq_len], device=positions.device, dtype=torch.long),
        context_lens=torch.tensor([past_seq_len], device=positions.device, dtype=torch.long),
        block_tables=block_table,
        slot_mapping=slot_mapping,
        block_size=seq_block_size,
        max_query_len=query_len,
        max_seq_len=seq_len,
        num_prefill_sequences=1 if query_len > 1 else 0,
        num_decode_tokens=1 if query_len == 1 else 0,
    )


def _resolve_device(configured_device: str) -> torch.device:
    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured_device)


def _resolve_dtype(configured_dtype: str) -> torch.dtype:
    if configured_dtype == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if configured_dtype not in dtype_map:
        raise ValueError(f"unsupported dtype={configured_dtype!r}")
    return dtype_map[configured_dtype]
