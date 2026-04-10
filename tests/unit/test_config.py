import pytest

from kvcore.api import EngineConfig, GenerationConfig, Request


def test_engine_config_validates_positive_values() -> None:
    with pytest.raises(ValueError):
        EngineConfig(max_new_tokens=0)

    with pytest.raises(ValueError):
        EngineConfig(block_size=0)


def test_generation_config_normalizes_eos_ids() -> None:
    config = GenerationConfig(eos_token_id=[1, 2, 3])

    assert config.normalized_eos_token_ids == (1, 2, 3)


def test_request_requires_prompt() -> None:
    with pytest.raises(ValueError):
        Request(prompt="")
