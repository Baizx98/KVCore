from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class TokenizerManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_model_source(
        cls,
        model: str,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> TokenizerManager:
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        return cls(tokenizer)

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        normalized_messages = [self._normalize_message(message) for message in messages]
        token_ids = self.tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if hasattr(token_ids, "input_ids"):
            token_ids = token_ids["input_ids"]
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if not isinstance(token_ids, list):
            raise TypeError(
                "tokenizer.apply_chat_template must return a token id list for a single "
                f"conversation, got {type(token_ids)!r}"
            )
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise TypeError("Batched chat template output is not supported in KVCore v1")
            token_ids = token_ids[0]
        return [int(token_id) for token_id in token_ids]

    def decode(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
        )

    def get_stop_token_ids(self) -> set[int]:
        stop_token_ids: set[int] = set()
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            stop_token_ids.add(int(eos_token_id))

        eos_token_ids = getattr(self.tokenizer, "eos_token_ids", None)
        if isinstance(eos_token_ids, int):
            stop_token_ids.add(int(eos_token_ids))
        elif eos_token_ids is not None:
            stop_token_ids.update(int(token_id) for token_id in eos_token_ids)
        return stop_token_ids

    @staticmethod
    def _normalize_message(message: Mapping[str, Any]) -> dict[str, str]:
        role = message.get("role")
        content = message.get("content")

        if role not in {"system", "user", "assistant"}:
            raise ValueError(
                "messages must use one of the roles "
                "'system', 'user', or 'assistant', "
                f"got {role!r}"
            )
        if not isinstance(content, str):
            raise TypeError(
                "message content must be a plain string in KVCore v1, "
                f"got {type(content)!r}"
            )
        return {"role": role, "content": content}


__all__ = [
    "TokenizerManager",
]
