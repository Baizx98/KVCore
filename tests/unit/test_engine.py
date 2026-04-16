from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from kvcore.api import Engine, EngineConfig, GenerationConfig, Request
from kvcore.api.types import GenerationResult
from kvcore.engine import LLMEngine
from kvcore.kv import KVManager
from kvcore.model_runner import ModelRunner
from kvcore.scheduler import Scheduler


class FakeTokenizer:
    eos_token_id = 3

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "|".join(str(token_id) for token_id in token_ids)


@dataclass
class FakeAdapter:
    tokenizer: FakeTokenizer
    num_hidden_layers: int = 2
    model_type: str = "llama"
    device: str = "cpu"
    _next_token: int = 1

    def encode_prompt(self, text: str) -> dict[str, FakeTensor]:
        return {
            "input_ids": FakeTensor([[10, 11, 12]]),
            "attention_mask": FakeTensor([[1, 1, 1]]),
        }

    def init_cache(self) -> FakeCache:
        return FakeCache()

    def prepare_layer_inputs(
        self,
        *,
        input_ids: FakeTensor,
        attention_mask: FakeTensor | None,
        past_key_values: FakeCache | None,
    ) -> FakePreparedInputs:
        cache = past_key_values or self.init_cache()
        q_len = len(input_ids.tolist())
        return FakePreparedInputs(
            hidden_states=FakeTensor([[0]]),
            causal_mask=attention_mask,
            position_ids=FakeTensor([[0]]),
            position_embeddings=("cos", "sin"),
            past_key_values=cache,
            past_seq_len=cache.get_seq_length(),
            q_len=q_len,
        )

    def iter_layers(self) -> tuple[FakeLayer, ...]:
        return (FakeLayer(), FakeLayer())

    def run_layer(
        self,
        *,
        layer_module: FakeLayer,
        hidden_states: FakeTensor,
        attention_mask: FakeTensor | None,
        position_ids: FakeTensor,
        position_embeddings: tuple[str, str],
        past_key_values: FakeCache,
        attention_params=None,
    ) -> FakeTensor:
        assert attention_params is not None
        assert attention_params.layer_id in {0, 1}
        assert attention_params.block_table.manager_block_ids
        layer_module.forward()
        past_key_values.seq_len = max(past_key_values.seq_len, position_ids.item() + 1)
        return hidden_states

    def before_attention(self, *, layer_module: FakeLayer, layer_context) -> None:
        layer_module.attention_params = layer_context.attention_params

    def after_attention(self, *, layer_module: FakeLayer, layer_context) -> None:
        layer_module.attention_params = None

    def finalize_hidden_states(self, hidden_states: FakeTensor) -> FakeTensor:
        return hidden_states

    def project_logits(self, hidden_states: FakeTensor) -> FakeTensor:
        token_id = self._next_token
        self._next_token = 3 if token_id == 1 else 0
        return FakeTensor.from_next_token(token_id)

    def decode_tokens(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def make_input_ids(self, token_ids: list[int], *, device: str) -> FakeTensor:
        return FakeTensor([token_ids])


class FakeLayer:
    attention_params = None

    def forward(self) -> None:
        return None


@dataclass
class FakePreparedInputs:
    hidden_states: FakeTensor
    causal_mask: FakeTensor | None
    position_ids: FakeTensor
    position_embeddings: tuple[str, str]
    past_key_values: FakeCache
    past_seq_len: int
    q_len: int


class FakeCache:
    def __init__(self) -> None:
        self.seq_len = 0

    def get_seq_length(self) -> int:
        return self.seq_len


class FakeTensor:
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_next_token(cls, token_id: int) -> FakeTensor:
        return cls(token_id)

    def __getitem__(self, item):
        if isinstance(self.data, int):
            return self
        if isinstance(item, tuple):
            return self
        return FakeTensor(self.data[item])

    def argmax(self, dim: int = -1) -> FakeTensor:
        return FakeTensor(self.data)

    def item(self) -> int:
        value = self.data
        while isinstance(value, list):
            value = value[0]
        return int(value)

    def tolist(self):
        if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
            return self.data[0]
        return self.data


@dataclass
class FakeLLMEngine:
    config: EngineConfig
    adapter: FakeAdapter
    result: GenerationResult

    def generate(
        self,
        request: Request,
        generation_config: GenerationConfig | None = None,
    ) -> GenerationResult:
        return self.result


def test_engine_delegates_to_llm_engine() -> None:
    expected = GenerationResult(
        text="ok",
        token_ids=[1, 2],
        generated_token_ids=[2],
        finish_reason="length",
        num_prompt_tokens=1,
        num_generated_tokens=1,
        request_id="req-test",
        kv_block_count=2,
        kv_total_tokens=2,
    )
    engine = Engine(
        config=EngineConfig(),
        llm_engine=cast(
            LLMEngine,
            FakeLLMEngine(
                config=EngineConfig(),
                adapter=FakeAdapter(tokenizer=FakeTokenizer()),
                result=expected,
            ),
        ),
    )

    result = engine.generate(Request(prompt="hello", request_id="req-test"))

    assert result is expected


def test_llm_engine_generate_uses_scheduler_kv_manager_and_model_runner() -> None:
    kv_manager = KVManager.from_model_config(num_layers=2, block_size=2, device="cpu")
    llm_engine = LLMEngine(
        config=EngineConfig(max_new_tokens=4, block_size=2),
        model_runner=ModelRunner(
            adapter=FakeAdapter(tokenizer=FakeTokenizer()),
            block_size=2,
            kv_manager=kv_manager,
        ),
        kv_manager=kv_manager,
        scheduler=Scheduler(),
    )

    result = llm_engine.generate(
        request=Request(prompt="hello", request_id="req-test"),
        generation_config=GenerationConfig(eos_token_id=3),
    )

    assert result.request_id == "req-test"
    assert result.num_prompt_tokens == 3
    assert result.num_generated_tokens >= 1
    assert result.finish_reason in {"eos", "length"}
    assert result.kv_total_tokens == result.num_prompt_tokens + result.num_generated_tokens
    assert len(kv_manager.layer_managers) == 2
