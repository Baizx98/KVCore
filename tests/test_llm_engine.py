from __future__ import annotations

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from kvcore.config import (
    CacheConfig,
    DeviceConfig,
    KVCoreConfig,
    ModelConfig,
    SchedulerConfig,
)
from kvcore.engine import engine_core as engine_core_module
from kvcore.entry.llm_engine import LLMEngine
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


def make_config(
    *,
    block_size: int = 2,
    num_gpu_blocks: int | None = 32,
    max_num_seqs: int = 2,
    max_num_scheduled_tokens: int = 8,
    max_model_len: int | None = 32,
    profile_kv_cache: bool = False,
    attn_backend: str = "torch_paged",
) -> KVCoreConfig:
    return KVCoreConfig(
        model_config=ModelConfig(
            model="unused",
            attn_backend=attn_backend,
            max_model_len=max_model_len,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            profile_kv_cache=profile_kv_cache,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
        ),
        device_config=DeviceConfig(device="cpu"),
    )


def make_tiny_model_runner(config: KVCoreConfig) -> ModelRunner:
    runner = ModelRunner(config)
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
            model_config=ModelConfig(
                model="unused",
                attn_backend=config.model_config.attn_backend,
                hf_config=hf_config,
            ),
            device_config=DeviceConfig(device="cpu"),
        )
    )
    return runner


def install_fake_engine_dependencies(monkeypatch, config: KVCoreConfig) -> None:
    monkeypatch.setattr(
        engine_core_module,
        "ModelRunner",
        lambda _config: make_tiny_model_runner(config),
    )
    monkeypatch.setattr(
        engine_core_module.TokenizerManager,
        "from_model_source",
        lambda **_kwargs: FakeTokenizerManager(),
    )


def test_llm_engine_generate_returns_text_and_token_ids(monkeypatch) -> None:
    config = make_config(max_model_len=32)
    install_fake_engine_dependencies(monkeypatch, config)
    engine = LLMEngine(config=config)

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


def test_llm_engine_accepts_kvcore_config(monkeypatch) -> None:
    config = make_config(block_size=2, num_gpu_blocks=32, max_model_len=16)
    install_fake_engine_dependencies(monkeypatch, config)
    engine = LLMEngine(config=config)

    assert engine.engine_core.config.model_config.model == "unused"
    assert engine.engine_core.config.cache_config.block_size == 2
    assert engine.engine_core.config.scheduler_config.max_num_seqs == 2


def test_llm_engine_single_request_torch_paged_e2e_outputs_text(monkeypatch) -> None:
    torch.manual_seed(0)
    config = make_config(max_num_seqs=1, max_num_scheduled_tokens=8, max_model_len=16)
    install_fake_engine_dependencies(monkeypatch, config)
    engine = LLMEngine(config=config)

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


def test_engine_profile_run_can_choose_cpu_kv_block_count(monkeypatch) -> None:
    config = make_config(
        block_size=4,
        num_gpu_blocks=None,
        profile_kv_cache=True,
        max_num_seqs=1,
        max_num_scheduled_tokens=4,
        max_model_len=32,
    )
    install_fake_engine_dependencies(monkeypatch, config)
    engine = LLMEngine(config=config)

    profile = engine.engine_core.kv_cache_profile
    assert profile.num_gpu_blocks == 17
    assert profile.max_tokens_per_sequence == 32
