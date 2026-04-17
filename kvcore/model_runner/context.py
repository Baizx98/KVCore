"""Execution context objects owned by the model runner package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kvcore.kv import BlockTable, LayerKVState, RequestKVView


@dataclass(slots=True)
class SequenceState:
    """Single-request sequence state tracked by the model runner."""

    request_id: str
    prompt_token_ids: list[int]
    kv_view: RequestKVView
    generated_token_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class AttentionParams:
    """Complete attention input description for one layer execution."""

    layer_id: int
    hidden_states: Any
    attention_mask: Any
    position_ids: Any
    position_embeddings: Any
    q_len: int
    kv_len: int
    kv_cache_seq_len_before: int
    kv_write_start: int
    kv_write_len: int
    is_prefill: bool
    block_size: int
    kv_cache_ref: str
    seq_block_ids: list[int]
    seq_block_starts: list[int]
    selected_block_ids: list[int]
    block_table: BlockTable
    kernel_block_ids: list[int]


@dataclass(slots=True)
class LayerContext:
    """Per-layer execution context exposed to the runner and hooks."""

    layer_id: int
    request_id: str
    layer_kv_state: LayerKVState
    attention_params: AttentionParams


@dataclass(slots=True)
class BatchContext:
    """Minimal batch context for the current single-request path."""

    request_id: str
    sequence_state: SequenceState
    layer_contexts: list[LayerContext]
    is_prefill: bool


@dataclass(slots=True)
class StepOutput:
    """Outputs produced by one explicit model-runner step."""

    logits: Any
    past_key_values: Any
    batch_context: BatchContext
