"""Llama adapter built on top of Hugging Face transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvcore.api.config import EngineConfig
from kvcore.logging import get_logger


@dataclass(slots=True)
class PreparedLayerInputs:
    """Inputs prepared for one explicit runner step."""

    hidden_states: Any
    causal_mask: Any
    position_ids: Any
    position_embeddings: Any
    past_key_values: Any
    past_seq_len: int
    q_len: int


@dataclass(slots=True)
class LlamaModelAdapter:
    """Thin wrapper around the Transformers Llama causal LM stack."""

    config: EngineConfig
    tokenizer: Any
    model: Any
    device: str
    torch_dtype: Any
    num_hidden_layers: int
    model_type: str

    @classmethod
    def from_config(cls, config: EngineConfig) -> LlamaModelAdapter:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "torch and transformers must be installed before loading a model"
            ) from exc

        logger = get_logger("model.llama")
        logger.info("loading tokenizer and model from %s", config.model_name_or_path)

        hf_config = AutoConfig.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        if hf_config.model_type != "llama":
            raise ValueError(f"expected a llama model, got model_type={hf_config.model_type!r}")

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = _resolve_device(config.device, torch)
        torch_dtype = _resolve_dtype(config.dtype, torch)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            dtype=torch_dtype,
        )
        model.eval()
        model.to(device)  # type: ignore[arg-type]

        logger.info(
            "loaded llama model_type=%s layers=%s device=%s dtype=%s",
            hf_config.model_type,
            hf_config.num_hidden_layers,
            device,
            str(torch_dtype),
        )

        return cls(
            config=config,
            tokenizer=tokenizer,
            model=model,
            device=device,
            torch_dtype=torch_dtype,
            num_hidden_layers=int(hf_config.num_hidden_layers),
            model_type=str(hf_config.model_type),
        )

    def encode_prompt(self, text: str) -> dict[str, Any]:
        encoded = self.tokenizer(text, return_tensors="pt")
        return {name: tensor.to(self.device) for name, tensor in encoded.items()}

    def prefill(self, input_ids: Any, attention_mask: Any) -> tuple[Any, Any]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values

    def decode_step(self, input_ids: Any, past_key_values: Any) -> tuple[Any, Any]:
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values

    def decode_tokens(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def make_input_ids(self, token_ids: list[int], *, device: str) -> Any:
        import torch

        return torch.tensor([token_ids], device=device)

    def init_cache(self) -> Any:
        from transformers.cache_utils import DynamicCache

        return DynamicCache(config=self.model.config)

    def prepare_layer_inputs(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None,
        past_key_values: Any | None,
    ) -> PreparedLayerInputs:
        from transformers.models.llama.modeling_llama import create_causal_mask

        inner_model = self.model.model
        cache = past_key_values if past_key_values is not None else self.init_cache()
        hidden_states = inner_model.embed_tokens(input_ids)
        past_seq_len = cache.get_seq_length()
        position_ids = self._build_position_ids(
            q_len=hidden_states.shape[1],
            device=hidden_states.device,
            past_seq_len=past_seq_len,
        )
        causal_mask = create_causal_mask(
            config=self.model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=cache,
            position_ids=position_ids,
        )
        position_embeddings = inner_model.rotary_emb(hidden_states, position_ids=position_ids)
        return PreparedLayerInputs(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=cache,
            past_seq_len=past_seq_len,
            q_len=int(hidden_states.shape[1]),
        )

    def iter_layers(self) -> tuple[Any, ...]:
        return tuple(self.model.model.layers[: self.model.config.num_hidden_layers])

    def run_layer(
        self,
        *,
        layer_module: Any,
        hidden_states: Any,
        attention_mask: Any,
        position_ids: Any,
        position_embeddings: Any,
        past_key_values: Any,
        attention_params: Any | None = None,
    ) -> Any:
        return layer_module(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            position_embeddings=position_embeddings,
            kvcore_attention_params=attention_params,
        )

    def before_attention(self, *, layer_module: Any, layer_context: Any) -> None:
        """Expose KVCore attention metadata to the HF Llama attention module."""

        attention_module = getattr(layer_module, "self_attn", None)
        if attention_module is None:
            return
        attention_module.kvcore_attention_params = layer_context.attention_params
        attention_module.kvcore_block_table = layer_context.attention_params.block_table

    def after_attention(self, *, layer_module: Any, layer_context: Any) -> None:
        """Remove per-step KVCore hook state after one layer finishes."""

        attention_module = getattr(layer_module, "self_attn", None)
        if attention_module is None:
            return
        for attr_name in ("kvcore_attention_params", "kvcore_block_table"):
            if hasattr(attention_module, attr_name):
                delattr(attention_module, attr_name)

    def finalize_hidden_states(self, hidden_states: Any) -> Any:
        return self.model.model.norm(hidden_states)

    def project_logits(self, hidden_states: Any) -> Any:
        return self.model.lm_head(hidden_states[:, -1:, :])

    def _build_position_ids(self, *, q_len: int, device: Any, past_seq_len: int) -> Any:
        import torch

        position_ids = torch.arange(q_len, device=device) + past_seq_len
        return position_ids.unsqueeze(0)


def _resolve_device(configured_device: str, torch: Any) -> str:
    if configured_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured_device


def _resolve_dtype(configured_dtype: str, torch: Any) -> Any:
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
