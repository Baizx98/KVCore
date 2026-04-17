from kvcore.model import Llama3Model, Mistral3Model, Qwen3Model, get_model_class


def test_model_registry_resolves_llama_family() -> None:
    assert get_model_class("llama") is Llama3Model


def test_model_registry_resolves_qwen_family_aliases() -> None:
    assert get_model_class("qwen3") is Qwen3Model
    assert get_model_class("qwen2") is Qwen3Model


def test_model_registry_resolves_mistral_family_aliases() -> None:
    assert get_model_class("mistral3") is Mistral3Model
    assert get_model_class("mistral") is Mistral3Model
