from __future__ import annotations

from dataclasses import dataclass

from kvcore.api import Engine, EngineConfig, GenerationConfig, Request


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

    def init_cache(self) -> object:
        return FakeCache()

    def prepare_layer_inputs(
        self,
        input_ids: FakeTensor,
        attention_mask: FakeTensor | None,
        past_key_values: FakeCache | None,
    ) -> FakePreparedInputs:
        cache = past_key_values or self.init_cache()
        q_len = len(input_ids.tolist())
        prepared = FakePreparedInputs(
            hidden_states=FakeTensor([[0]]),
            causal_mask=attention_mask,
            position_ids=FakeTensor([[0]]),
            position_embeddings=("cos", "sin"),
            past_key_values=cache,
            past_seq_len=cache.get_seq_length(),
            q_len=q_len,
        )
        return prepared

    def iter_layers(self) -> tuple[FakeLayer, ...]:
        return (FakeLayer(), FakeLayer())

    def run_layer(
        self,
        *,
        layer_module: "FakeLayer",
        hidden_states: FakeTensor,
        attention_mask: FakeTensor | None,
        position_ids: FakeTensor,
        position_embeddings: tuple[str, str],
        past_key_values: "FakeCache",
    ) -> FakeTensor:
        layer_module.forward()
        return hidden_states

    def finalize_hidden_states(self, hidden_states: FakeTensor) -> FakeTensor:
        return hidden_states

    def project_logits(self, hidden_states: FakeTensor) -> FakeTensor:
        token_id = self._next_token
        self._next_token = 3 if token_id == 1 else 0
        return FakeTensor.from_next_token(token_id)

    def decode_tokens(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class FakeLayer:
    def forward(self) -> None:
        return None


@dataclass
class FakePreparedInputs:
    hidden_states: "FakeTensor"
    causal_mask: "FakeTensor | None"
    position_ids: "FakeTensor"
    position_embeddings: tuple[str, str]
    past_key_values: "FakeCache"
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
        data = self.data
        if isinstance(item, tuple):
            return self
        return FakeTensor(data[item])

    def argmax(self, dim: int = -1) -> FakeTensor:
        return FakeTensor(self.data)

    def item(self) -> int:
        if isinstance(self.data, list):
            value = self.data
            while isinstance(value, list):
                value = value[0]
            return int(value)
        return int(self.data)

    def tolist(self):
        if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
            return self.data[0]
        return self.data


class FakeTorchModule:
    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    @staticmethod
    def no_grad():
        return FakeTorchModule._NoGrad()

    @staticmethod
    def tensor(data, device: str = "cpu") -> FakeTensor:
        return FakeTensor(data)


def test_engine_generate_uses_greedy_runtime(monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorchModule)

    engine = Engine(
        config=EngineConfig(max_new_tokens=4, block_size=2),
        adapter=FakeAdapter(tokenizer=FakeTokenizer()),
    )
    result = engine.generate(
        request=Request(prompt="hello", request_id="req-test"),
        generation_config=GenerationConfig(eos_token_id=3),
    )

    assert result.request_id == "req-test"
    assert result.num_prompt_tokens == 3
    assert result.num_generated_tokens >= 1
    assert result.finish_reason in {"eos", "length"}
    assert result.kv_total_tokens == result.num_prompt_tokens + result.num_generated_tokens
