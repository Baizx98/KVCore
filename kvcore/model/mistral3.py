"""Mistral 3 model implementation with manual layers."""

from __future__ import annotations

from kvcore.model.base import HFDecoderModel
from kvcore.model.llama3 import LlamaForCausalLM


class Mistral3Model(HFDecoderModel):
    """Manual Mistral-family model wrapper."""

    family_name = "mistral3"
    supported_model_types = ("mistral3", "mistral")
    model_cls = LlamaForCausalLM
