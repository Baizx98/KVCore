from __future__ import annotations

import torch
from torch import nn


class LogitsProcessor(nn.Module):
    """Compute final logits from hidden states or already-computed logits.

    This mirrors the small, model-adjacent part of vLLM's logits processor:
    project hidden states with ``lm_head`` when needed, optionally apply soft
    capping, then apply a global logit scale. Sampling-time filtering is kept
    in ``Sampler`` so the model and runtime boundaries stay clean.
    """

    def __init__(
        self,
        *,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        super().__init__()
        if soft_cap is not None and soft_cap <= 0:
            raise ValueError(f"soft_cap must be positive when set, got {soft_cap}")
        self.scale = scale
        self.logits_as_input = logits_as_input
        self.soft_cap = soft_cap

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module | None = None,
        *,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.logits_as_input:
            logits = hidden_states
        else:
            if lm_head is None:
                raise ValueError("lm_head is required when logits_as_input=False")
            logits = lm_head(hidden_states)
            if embedding_bias is not None:
                logits = logits + embedding_bias

        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale
        return logits

    def extra_repr(self) -> str:
        return (
            f"scale={self.scale}, logits_as_input={self.logits_as_input}, "
            f"soft_cap={self.soft_cap}"
        )
