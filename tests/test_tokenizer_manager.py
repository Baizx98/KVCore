from __future__ import annotations

from pathlib import Path

import pytest

from kvcore.utils.tokenizer import TokenizerManager


def test_tokenizer_manager_rejects_non_string_message_content() -> None:
    with pytest.raises(TypeError):
        TokenizerManager._normalize_message({"role": "user", "content": ["bad"]})  # type: ignore[arg-type]


@pytest.mark.skipif(
    not Path("/Tan/model/Llama-3.1-8B-Instruct").exists(),
    reason="Local Llama-3.1-8B-Instruct tokenizer not available",
)
def test_tokenizer_manager_loads_local_llama31_chat_template() -> None:
    manager = TokenizerManager.from_model_source("/Tan/model/Llama-3.1-8B-Instruct")
    token_ids = manager.encode_messages(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello."},
        ]
    )

    assert token_ids
    decoded = manager.decode(token_ids)
    assert isinstance(decoded, str)
