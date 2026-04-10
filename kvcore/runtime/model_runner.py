"""Single-request explicit layer-wise model runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvcore.kv import RequestKVView
from kvcore.logging import get_logger
from kvcore.runtime.context import (
    AttentionParams,
    BatchContext,
    LayerContext,
    SequenceState,
    StepOutput,
)


@dataclass(slots=True)
class SingleRequestModelRunner:
    """Run one request through an explicit layer-by-layer reference path."""

    adapter: Any
    block_size: int

    def initialize_sequence_state(
        self,
        *,
        request_id: str,
        prompt_token_ids: list[int],
        kv_view: RequestKVView,
    ) -> SequenceState:
        return SequenceState(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            kv_view=kv_view,
        )

    def run_prefill(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        sequence_state: SequenceState,
    ) -> StepOutput:
        return self._run_step(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            sequence_state=sequence_state,
            is_prefill=True,
        )

    def run_decode_step(
        self,
        *,
        input_ids: Any,
        past_key_values: Any,
        sequence_state: SequenceState,
    ) -> StepOutput:
        return self._run_step(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=past_key_values,
            sequence_state=sequence_state,
            is_prefill=False,
        )

    def before_layer(self, layer_context: LayerContext) -> None:
        """Future hook point before one layer executes."""
        return None

    def after_layer(self, layer_context: LayerContext) -> None:
        """Future hook point after one layer executes."""
        return None

    def _run_step(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        past_key_values: Any,
        sequence_state: SequenceState,
        is_prefill: bool,
    ) -> StepOutput:
        logger = get_logger("runtime.runner")
        prepared_inputs = self.adapter.prepare_layer_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        hidden_states = prepared_inputs.hidden_states
        cache = prepared_inputs.past_key_values
        past_seq_len = prepared_inputs.past_seq_len
        q_len = prepared_inputs.q_len
        kv_len = past_seq_len + q_len
        layer_contexts: list[LayerContext] = []

        for layer_id, layer_module in enumerate(self.adapter.iter_layers()):
            layer_kv_state = sequence_state.kv_view.layer_states[layer_id]
            attn_params = self._build_attention_params(
                hidden_states=hidden_states,
                layer_id=layer_id,
                sequence_state=sequence_state,
                prepared_inputs=prepared_inputs,
                past_seq_len=past_seq_len,
                kv_len=kv_len,
                q_len=q_len,
                is_prefill=is_prefill,
            )
            layer_context = LayerContext(
                layer_id=layer_id,
                request_id=sequence_state.request_id,
                layer_kv_state=layer_kv_state,
                attention_params=attn_params,
            )
            self.before_layer(layer_context)
            hidden_states = self.adapter.run_layer(
                layer_module=layer_module,
                hidden_states=hidden_states,
                attention_mask=prepared_inputs.causal_mask,
                position_ids=prepared_inputs.position_ids,
                position_embeddings=prepared_inputs.position_embeddings,
                past_key_values=cache,
            )
            self.after_layer(layer_context)
            layer_contexts.append(layer_context)

        hidden_states = self.adapter.finalize_hidden_states(hidden_states)
        logits = self.adapter.project_logits(hidden_states)
        logger.debug(
            "request_id=%s is_prefill=%s q_len=%s kv_len=%s layers=%s",
            sequence_state.request_id,
            is_prefill,
            q_len,
            kv_len,
            len(layer_contexts),
        )
        return StepOutput(
            logits=logits,
            past_key_values=cache,
            batch_context=BatchContext(
                request_id=sequence_state.request_id,
                sequence_state=sequence_state,
                layer_contexts=layer_contexts,
                is_prefill=is_prefill,
            ),
        )

    def _build_attention_params(
        self,
        *,
        hidden_states: Any,
        layer_id: int,
        sequence_state: SequenceState,
        prepared_inputs: Any,
        past_seq_len: int,
        kv_len: int,
        q_len: int,
        is_prefill: bool,
    ) -> AttentionParams:
        layer_state = sequence_state.kv_view.layer_states[layer_id]
        seq_block_ids = [block.block_id for block in layer_state.blocks]
        seq_block_starts = [block.token_start for block in layer_state.blocks]
        selected_block_ids = list(seq_block_ids)
        canonical_ref = sequence_state.kv_view.canonical_layer_states[layer_id].ref
        return AttentionParams(
            layer_id=layer_id,
            hidden_states=hidden_states,
            attention_mask=prepared_inputs.causal_mask,
            position_ids=prepared_inputs.position_ids,
            position_embeddings=prepared_inputs.position_embeddings,
            q_len=q_len,
            kv_len=kv_len,
            kv_cache_seq_len_before=past_seq_len,
            kv_write_start=past_seq_len,
            kv_write_len=q_len,
            is_prefill=is_prefill,
            block_size=self.block_size,
            kv_cache_ref=canonical_ref.tensor_name,
            seq_block_ids=seq_block_ids,
            seq_block_starts=seq_block_starts,
            selected_block_ids=selected_block_ids,
        )
