from __future__ import annotations

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from kvcore.config import KVCoreConfig, ModelConfig, RuntimeConfig
from kvcore.engine.engine_core import EngineConfig
from kvcore.entry.llm_engine import LLMEngine
from kvcore.model.model_loader import ModelLoadConfig
from kvcore.model.model_runner import ModelRunner
from kvcore.model.models.llama3 import Llama3ForCausalLM
from kvcore.utils.sampling_params import SamplingParams


class FakeTokenizerManager:
    def encode_messages(self, messages, *, add_generation_prompt: bool = True) -> list[int]:
        del add_generation_prompt
        text = " ".join(str(message["content"]) for message in messages)
        return [len(text) % 10 + 1, 2, 3]

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(str(token_id) for token_id in token_ids)

    def get_stop_token_ids(self) -> set[int]:
        return set()


def make_tiny_model_runner(attn_backend: str = "torch_paged") -> ModelRunner:
    load_config = ModelLoadConfig(model="unused", device="cpu", attn_backend=attn_backend)
    runner = ModelRunner(load_config)
    hf_config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    runner.model = Llama3ForCausalLM(
        KVCoreConfig(
            model=ModelConfig(
                model="unused",
                device="cpu",
                attn_backend=attn_backend,
                hf_config=hf_config,
            )
        )
    )
    return runner


def test_llm_engine_generate_returns_text_and_token_ids() -> None:
    engine = LLMEngine(
        load_config=ModelLoadConfig(model="unused", device="cpu", attn_backend="torch_paged"),
        engine_config=EngineConfig(
            block_size=2,
            num_gpu_blocks=32,
            max_num_seqs=2,
            max_num_scheduled_tokens=8,
            max_model_len=32,
        ),
        model_runner=make_tiny_model_runner(),
        tokenizer_manager=FakeTokenizerManager(),
    )

    outputs = engine.generate(
        [
            {
                "request_id": "req-1",
                "messages": [{"role": "user", "content": "Hello"}],
                "sampling_params": SamplingParams(max_tokens=2, temperature=0.0),
            }
        ]
    )

    assert len(outputs) == 1
    assert outputs[0].request_id == "req-1"
    assert len(outputs[0].output_token_ids) == 2
    assert isinstance(outputs[0].output_text, str)
    assert outputs[0].finish_reason == "length"


def test_llm_engine_accepts_kvcore_config() -> None:
    engine = LLMEngine(
        config=KVCoreConfig(
            model=ModelConfig(model="unused", device="cpu", attn_backend="torch_paged"),
            runtime=RuntimeConfig(
                block_size=2,
                num_gpu_blocks=32,
                max_model_len=16,
            ),
        ),
        model_runner=make_tiny_model_runner("torch_paged"),
        tokenizer_manager=FakeTokenizerManager(),
    )

    assert engine.engine_core.config.model.model == "unused"
    assert engine.engine_core.config.runtime.block_size == 2
    assert engine.engine_core.config.scheduler.max_num_seqs == 8


def test_legacy_engine_config_is_normalized_to_kvcore_config() -> None:
    engine = LLMEngine(
        load_config=ModelLoadConfig(model="unused", device="cpu", attn_backend="torch_paged"),
        engine_config=EngineConfig(
            block_size=4,
            num_gpu_blocks=16,
            max_num_seqs=3,
            max_num_scheduled_tokens=7,
            max_model_len=16,
        ),
        model_runner=make_tiny_model_runner("torch_paged"),
        tokenizer_manager=FakeTokenizerManager(),
    )

    assert engine.engine_core.config.runtime.block_size == 4
    assert engine.engine_core.config.scheduler.max_num_seqs == 3
    assert engine.engine_core.config.scheduler.max_num_scheduled_tokens == 7


def test_llm_engine_single_request_torch_paged_e2e_outputs_text() -> None:
    torch.manual_seed(0)
    engine = LLMEngine(
        load_config=ModelLoadConfig(model="unused", device="cpu", attn_backend="torch_paged"),
        engine_config=EngineConfig(
            block_size=2,
            num_gpu_blocks=32,
            max_num_seqs=1,
            max_num_scheduled_tokens=8,
            max_model_len=16,
        ),
        model_runner=make_tiny_model_runner("torch_paged"),
        tokenizer_manager=FakeTokenizerManager(),
    )

    outputs = engine.generate(
        [
            {
                "request_id": "paged-req",
                "messages": [{"role": "user", "content": "Hello paged KV"}],
                "sampling_params": SamplingParams(max_tokens=2, temperature=0.0),
            }
        ]
    )

    assert len(outputs) == 1
    assert outputs[0].request_id == "paged-req"
    assert len(outputs[0].output_token_ids) == 2
    assert outputs[0].output_text
    assert outputs[0].finish_reason == "length"
    print(f"single request torch_paged inference content: {outputs[0].output_text}")


def test_engine_profile_run_can_choose_cpu_kv_block_count() -> None:
    runner = make_tiny_model_runner("torch_paged")
    engine = LLMEngine(
        load_config=ModelLoadConfig(model="unused", device="cpu", attn_backend="torch_paged"),
        engine_config=EngineConfig(
            block_size=4,
            num_gpu_blocks=None,
            profile_kv_cache=True,
            max_num_seqs=1,
            max_num_scheduled_tokens=4,
            max_model_len=32,
        ),
        model_runner=runner,
        tokenizer_manager=FakeTokenizerManager(),
    )

    profile = engine.engine_core.kv_cache_profile
    assert profile.num_gpu_blocks == 17
    assert profile.max_tokens_per_sequence == 32
