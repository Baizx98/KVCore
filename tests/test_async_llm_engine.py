from __future__ import annotations

import asyncio

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from kvcore.config import CacheConfig, DeviceConfig, KVCoreConfig, ModelConfig, SchedulerConfig
from kvcore.engine import engine_core as engine_core_module
from kvcore.entry.async_llm_engine import AsyncLLMEngine
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


def make_config() -> KVCoreConfig:
    return KVCoreConfig(
        model_config=ModelConfig(model="unused", attn_backend="torch_paged", max_model_len=32),
        cache_config=CacheConfig(block_size=2, num_gpu_blocks=32),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_scheduled_tokens=8),
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
                attn_backend="torch_paged",
                hf_config=hf_config,
            ),
            device_config=DeviceConfig(device="cpu"),
        )
    )
    return runner


def make_async_engine() -> AsyncLLMEngine:
    config = make_config()
    return AsyncLLMEngine(config=config)


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


def test_async_llm_engine_generate_offline_batch(monkeypatch) -> None:
    async def run() -> None:
        install_fake_engine_dependencies(monkeypatch, make_config())
        engine = make_async_engine()
        outputs = await engine.generate(
            [
                {
                    "request_id": "req-1",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "sampling_params": SamplingParams(max_tokens=1, temperature=0.0),
                },
                {
                    "request_id": "req-2",
                    "messages": [{"role": "user", "content": "World"}],
                    "sampling_params": SamplingParams(max_tokens=1, temperature=0.0),
                },
            ]
        )
        assert {output.request_id for output in outputs} == {"req-1", "req-2"}
        assert all(len(output.output_token_ids) == 1 for output in outputs)

    asyncio.run(run())


def test_async_llm_engine_submit_simulated_online_requests(monkeypatch) -> None:
    async def run() -> None:
        install_fake_engine_dependencies(monkeypatch, make_config())
        engine = make_async_engine()
        first = await engine.submit(
            "req-1",
            [{"role": "user", "content": "first"}],
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        await engine.run_until_idle()
        second = await engine.submit(
            "req-2",
            [{"role": "user", "content": "second"}],
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        await engine.run_until_idle()

        assert (await first).request_id == "req-1"
        assert (await second).request_id == "req-2"

    asyncio.run(run())


def test_async_llm_engine_run_until_idle_with_no_requests(monkeypatch) -> None:
    async def run() -> None:
        install_fake_engine_dependencies(monkeypatch, make_config())
        engine = make_async_engine()
        await engine.run_until_idle()
        assert not engine.engine_core.has_unfinished_requests()

    asyncio.run(run())
