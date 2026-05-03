from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from kvcore.config import KVCoreConfig
from kvcore.kv.kv_manager import KVManagerConfig
from kvcore.kv.single_type_kv_manager import KVLayerSpec
from kvcore.model.forward_context import ForwardContext, set_forward_context
from kvcore.model.input_batch import CachedRequestState, InputBatch
from kvcore.model.kv_runtime import PagedAttentionMetadata
from kvcore.model.model_loader import DefaultModelLoader
from kvcore.sample import Sampler
from kvcore.sched.utils import SchedulerOutput
from kvcore.utils.log import get_logger
from kvcore.utils.sampling_params import SamplingParams

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class KVCacheProfileResult:
    num_gpu_blocks: int
    block_size: int
    bytes_per_block: int
    available_memory_bytes: int | None
    memory_budget_bytes: int | None
    max_tokens_per_sequence: int


@dataclass(frozen=True, slots=True)
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]

    @classmethod
    def empty(cls) -> ModelRunnerOutput:
        return cls(req_ids=[], req_id_to_index={}, sampled_token_ids=[])


@dataclass(frozen=True, slots=True)
class ExecuteModelState:
    scheduler_output: SchedulerOutput
    logits: torch.Tensor
    sampled_request_ids: tuple[str, ...]
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampling_params: tuple[SamplingParams, ...]
    num_reqs: int
    num_scheduled_tokens: int
    num_prefill_reqs: int
    num_decode_reqs: int
    prepare_time_sec: float
    metadata_time_sec: float
    forward_time_sec: float
    num_zeroed_blocks: int


@dataclass(frozen=True, slots=True)
class ModelRunnerStepStats:
    num_reqs: int
    num_scheduled_tokens: int
    num_prefill_reqs: int
    num_decode_reqs: int
    prepare_time_sec: float
    metadata_time_sec: float
    forward_time_sec: float
    sample_time_sec: float
    num_zeroed_blocks: int


class ModelRunner:
    """Owns model creation, worker-local request state, and model execution."""

    def __init__(self, kvcore_config: KVCoreConfig) -> None:
        self.kvcore_config = kvcore_config
        self.model_loader = DefaultModelLoader(kvcore_config)
        self.model: nn.Module | None = None
        self.kv_manager_config: KVManagerConfig | None = None
        self.kv_cache_tensor: torch.Tensor | None = None
        self.input_batch: InputBatch | None = None
        self.requests: dict[str, CachedRequestState] = {}
        self.sampler = Sampler()
        self.execute_model_state: ExecuteModelState | None = None
        self.last_step_stats: ModelRunnerStepStats | None = None

        self.input_ids: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None
        self.query_start_loc: torch.Tensor | None = None
        self.seq_lens: torch.Tensor | None = None
        self.context_lens: torch.Tensor | None = None
        self.query_lens: torch.Tensor | None = None
        self.token_request_indices: torch.Tensor | None = None

        logger.info(
            "ModelRunner initialized model=%s device=%s attn_backend=%s",
            self.kvcore_config.model_config.model,
            self.kvcore_config.device_config.device,
            self.kvcore_config.model_config.attn_backend,
        )

    def load_model(self) -> nn.Module:
        logger.info("Loading model model=%s", self.kvcore_config.model_config.model)
        self.model = self.model_loader.load_model()
        logger.info("Model loaded model=%s", self.kvcore_config.model_config.model)
        return self.model

    def profile_run(
        self,
        *,
        layer_specs: tuple[KVLayerSpec, ...],
        block_size: int,
        max_model_len: int,
        requested_num_gpu_blocks: int | None,
        should_profile: bool,
        gpu_memory_utilization: float,
    ) -> KVCacheProfileResult:
        """Resolve KV block capacity from runner-owned model/device state."""
        if requested_num_gpu_blocks is None or should_profile:
            num_gpu_blocks = self._estimate_num_gpu_blocks(
                layer_specs=layer_specs,
                block_size=block_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        else:
            num_gpu_blocks = requested_num_gpu_blocks
        logger.info(
            "ModelRunner profile resolved num_gpu_blocks=%d requested=%s "
            "should_profile=%s",
            num_gpu_blocks,
            requested_num_gpu_blocks,
            should_profile,
        )
        return self._build_kv_cache_profile_result(
            layer_specs=layer_specs,
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def initialize_kv_cache(self, kv_manager_config: KVManagerConfig) -> torch.Tensor:
        self.kv_manager_config = kv_manager_config
        self.kv_cache_tensor = self.initialize_kv_cache_tensor(kv_manager_config)
        self.input_batch = self._make_input_batch(kv_manager_config)
        self._init_step_buffers()
        logger.info(
            "Initialized KV cache tensor shape=%s dtype=%s device=%s",
            tuple(self.kv_cache_tensor.shape),
            self.kv_cache_tensor.dtype,
            self.kv_cache_tensor.device,
        )
        return self.kv_cache_tensor

    def initialize_kv_cache_tensor(
        self,
        kv_manager_config: KVManagerConfig,
    ) -> torch.Tensor:
        device = self._resolve_device()
        first_spec = kv_manager_config.layer_specs[0]
        dtype = first_spec.dtype if isinstance(first_spec.dtype, torch.dtype) else torch.float16
        for layer_spec in kv_manager_config.layer_specs[1:]:
            if (
                layer_spec.block_size != first_spec.block_size
                or layer_spec.num_kv_heads != first_spec.num_kv_heads
                or layer_spec.head_size != first_spec.head_size
                or layer_spec.dtype != first_spec.dtype
            ):
                raise NotImplementedError(
                    "Shared KV cache tensor currently requires uniform layer specs. "
                    "Non-uniform layers need explicit offset/stride metadata."
                )
        return torch.empty(
            (
                2,
                kv_manager_config.num_gpu_blocks,
                first_spec.block_size,
                first_spec.num_kv_heads,
                first_spec.head_size,
            ),
            dtype=dtype,
            device=device,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | None:
        if self.execute_model_state is not None:
            raise RuntimeError("sample_tokens() must be called after execute_model()")

        self.last_step_stats = None
        self._update_states(scheduler_output)
        if scheduler_output.is_empty or scheduler_output.total_num_scheduled_tokens == 0:
            self.last_step_stats = ModelRunnerStepStats(0, 0, 0, 0, 0, 0, 0, 0, 0)
            logger.debug("ModelRunner skipped zero-token scheduler output")
            return ModelRunnerOutput.empty()

        num_zeroed_blocks = self._zero_new_blocks(scheduler_output.new_block_ids_to_zero)
        input_batch = self._require_input_batch()
        req_ids = input_batch.req_ids
        num_scheduled_tokens = tuple(
            scheduler_output.num_scheduled_tokens[req_id] for req_id in req_ids
        )

        prepare_start = perf_counter()
        (
            logits_indices,
            sampled_request_ids,
            sampling_params,
            num_prefill_reqs,
            num_decode_reqs,
            max_query_len,
            max_seq_len,
        ) = self._prepare_inputs(scheduler_output, num_scheduled_tokens)
        prepare_time = perf_counter() - prepare_start

        metadata_start = perf_counter()
        attn_metadata = self._build_attention_metadata(
            num_reqs=len(req_ids),
            num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
        )
        metadata_time = perf_counter() - metadata_start

        forward_start = perf_counter()
        hidden_states = self._run_forward(
            attn_metadata,
            scheduler_output.total_num_scheduled_tokens,
        )
        forward_time = perf_counter() - forward_start

        logits = self._compute_logits(hidden_states, logits_indices)
        self.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            logits=logits,
            sampled_request_ids=sampled_request_ids,
            req_ids=list(req_ids),
            req_id_to_index=dict(input_batch.req_id_to_index),
            sampling_params=sampling_params,
            num_reqs=len(req_ids),
            num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
            prepare_time_sec=prepare_time,
            metadata_time_sec=metadata_time,
            forward_time_sec=forward_time,
            num_zeroed_blocks=num_zeroed_blocks,
        )
        logger.info(
            "ModelRunner forward reqs=%d tokens=%d prefill=%d decode=%d "
            "samples=%d zeroed_blocks=%d prepare=%.6fs metadata=%.6fs "
            "forward=%.6fs",
            len(req_ids),
            scheduler_output.total_num_scheduled_tokens,
            num_prefill_reqs,
            num_decode_reqs,
            len(sampled_request_ids),
            num_zeroed_blocks,
            prepare_time,
            metadata_time,
            forward_time,
        )
        return None

    @torch.inference_mode()
    def sample_tokens(self) -> ModelRunnerOutput:
        state = self.execute_model_state
        if state is None:
            return ModelRunnerOutput.empty()
        self.execute_model_state = None

        sample_start = perf_counter()
        sampled_by_req_id: dict[str, int] = {}
        if state.sampled_request_ids:
            sampled_token_ids = self.sampler.sample(state.logits, state.sampling_params)
            sampled_tuple = tuple(int(token_id) for token_id in sampled_token_ids.tolist())
            self._require_input_batch().record_sampled_tokens(
                state.sampled_request_ids,
                sampled_tuple,
            )
            sampled_by_req_id = dict(
                zip(state.sampled_request_ids, sampled_tuple, strict=True)
            )
            logger.debug(
                "Sampled tokens request_ids=%s token_ids=%s",
                state.sampled_request_ids,
                sampled_tuple,
            )
        sample_time = perf_counter() - sample_start

        self.last_step_stats = ModelRunnerStepStats(
            num_reqs=state.num_reqs,
            num_scheduled_tokens=state.num_scheduled_tokens,
            num_prefill_reqs=state.num_prefill_reqs,
            num_decode_reqs=state.num_decode_reqs,
            prepare_time_sec=state.prepare_time_sec,
            metadata_time_sec=state.metadata_time_sec,
            forward_time_sec=state.forward_time_sec,
            sample_time_sec=sample_time,
            num_zeroed_blocks=state.num_zeroed_blocks,
        )
        return ModelRunnerOutput(
            req_ids=state.req_ids,
            req_id_to_index=state.req_id_to_index,
            sampled_token_ids=[
                [sampled_by_req_id[req_id]] if req_id in sampled_by_req_id else []
                for req_id in state.req_ids
            ],
        )

    def remove_requests(self, request_ids: tuple[str, ...] | list[str]) -> None:
        for request_id in request_ids:
            self.requests.pop(request_id, None)
        self._require_input_batch().remove_requests(request_ids)

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        input_batch = self._require_input_batch()
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            input_batch.remove_request(req_id)

        scheduled_req_ids = set(scheduler_output.num_scheduled_tokens)
        unscheduled_req_ids = set(input_batch.req_id_to_index) - scheduled_req_ids
        for req_id in unscheduled_req_ids:
            input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        for new_request in scheduler_output.scheduled_new_reqs:
            request_state = CachedRequestState(
                req_id=new_request.req_id,
                prompt_token_ids=list(new_request.prompt_token_ids),
                sampling_params=new_request.sampling_params,
                block_ids=new_request.block_ids,
                num_computed_tokens=new_request.num_computed_tokens,
            )
            self.requests[new_request.req_id] = request_state
            reqs_to_add.append(request_state)

        cached_requests = scheduler_output.scheduled_cached_reqs
        for req_index, req_id in enumerate(cached_requests.req_ids):
            request_state = self.requests[req_id]
            row_index = input_batch.req_id_to_index.get(req_id)
            if row_index is None:
                self._maybe_extend_cached_request(
                    request_state,
                    cached_requests.new_token_ids[req_index],
                    cached_requests.num_computed_tokens[req_index],
                )
                request_state.block_ids = cached_requests.block_ids[req_index]
                request_state.num_computed_tokens = cached_requests.num_computed_tokens[
                    req_index
                ]
                reqs_to_add.append(request_state)
                continue
            input_batch.update_cached_request(
                req_id=req_id,
                new_token_ids=cached_requests.new_token_ids[req_index],
                block_ids=cached_requests.block_ids[req_index],
                num_computed_tokens=cached_requests.num_computed_tokens[req_index],
            )

        for request_state in reqs_to_add:
            input_batch.add_request(request_state)
        input_batch.condense()

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: tuple[int, ...],
    ) -> tuple[torch.Tensor, tuple[str, ...], tuple[SamplingParams, ...], int, int, int, int]:
        input_batch = self._require_input_batch()
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_reqs = input_batch.num_reqs
        if total_num_scheduled_tokens <= 0 or num_reqs <= 0:
            raise ValueError("Cannot prepare model inputs for an empty batch")

        flat_input_token_ids: list[int] = []
        flat_positions: list[int] = []
        query_start_loc = [0]
        seq_lens: list[int] = []
        context_lens: list[int] = []
        query_lens: list[int] = []
        token_request_indices: list[int] = []
        logits_indices: list[int] = []
        sampled_request_ids: list[str] = []
        sampling_params: list[SamplingParams] = []
        num_prefill_reqs = 0
        num_decode_reqs = 0

        for request_idx, (req_id, num_tokens) in enumerate(
            zip(input_batch.req_ids, num_scheduled_tokens, strict=True)
        ):
            row_idx = input_batch.req_id_to_index[req_id]
            start = int(input_batch.num_computed_tokens_cpu[row_idx])
            end = start + num_tokens
            available = int(input_batch.num_tokens[row_idx])
            if end > available:
                raise ValueError(
                    "Scheduler requested tokens that are not present in InputBatch: "
                    f"req_id={req_id} start={start} end={end} available={available}"
                )
            flat_input_token_ids.extend(
                int(token_id)
                for token_id in input_batch.token_ids_cpu[row_idx, start:end]
            )
            flat_positions.extend(range(start, end))
            query_start_loc.append(query_start_loc[-1] + num_tokens)
            seq_lens.append(end)
            context_lens.append(start)
            query_lens.append(num_tokens)
            token_request_indices.extend([request_idx] * num_tokens)
            if start < int(input_batch.num_prompt_tokens[row_idx]):
                num_prefill_reqs += 1
            else:
                num_decode_reqs += 1
            if end >= available:
                logits_indices.append(query_start_loc[-1] - 1)
                sampled_request_ids.append(req_id)
                sampling_params.append(input_batch.get_request(req_id).sampling_params)

        self._require_step_buffers()
        assert self.input_ids is not None
        assert self.positions is not None
        assert self.query_start_loc is not None
        assert self.seq_lens is not None
        assert self.context_lens is not None
        assert self.query_lens is not None
        assert self.token_request_indices is not None

        self.input_ids[:total_num_scheduled_tokens] = torch.tensor(
            flat_input_token_ids,
            dtype=torch.long,
            device=self.input_ids.device,
        )
        self.positions[:total_num_scheduled_tokens] = torch.tensor(
            flat_positions,
            dtype=torch.long,
            device=self.positions.device,
        )
        self.query_start_loc[: num_reqs + 1] = torch.tensor(
            query_start_loc,
            dtype=torch.int32,
            device=self.query_start_loc.device,
        )
        self.seq_lens[:num_reqs] = torch.tensor(
            seq_lens,
            dtype=torch.int32,
            device=self.seq_lens.device,
        )
        self.context_lens[:num_reqs] = torch.tensor(
            context_lens,
            dtype=torch.int32,
            device=self.context_lens.device,
        )
        self.query_lens[:num_reqs] = torch.tensor(
            query_lens,
            dtype=torch.int32,
            device=self.query_lens.device,
        )
        self.token_request_indices[:total_num_scheduled_tokens] = torch.tensor(
            token_request_indices,
            dtype=torch.int32,
            device=self.token_request_indices.device,
        )

        logger.debug(
            "Prepared model input reqs=%d tokens=%d samples=%d max_query_len=%d "
            "max_seq_len=%d",
            num_reqs,
            total_num_scheduled_tokens,
            len(logits_indices),
            max(query_lens, default=0),
            max(seq_lens, default=0),
        )
        return (
            torch.tensor(logits_indices, dtype=torch.long, device=self._resolve_device()),
            tuple(sampled_request_ids),
            tuple(sampling_params),
            num_prefill_reqs,
            num_decode_reqs,
            max(query_lens, default=0),
            max(seq_lens, default=0),
        )

    def _build_attention_metadata(
        self,
        *,
        num_reqs: int,
        num_scheduled_tokens: int,
        num_prefill_reqs: int,
        num_decode_reqs: int,
        max_query_len: int,
        max_seq_len: int,
    ) -> PagedAttentionMetadata:
        input_batch = self._require_input_batch()
        kv_cache_tensor = self._require_kv_cache_tensor()
        self._require_step_buffers()
        assert self.positions is not None
        assert self.query_start_loc is not None
        assert self.seq_lens is not None
        assert self.context_lens is not None
        assert self.query_lens is not None
        assert self.token_request_indices is not None

        input_batch.block_table.commit_block_table(num_reqs)
        input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc[: num_reqs + 1],
            self.positions[:num_scheduled_tokens].to(dtype=torch.int32),
        )
        slot_mapping = {
            layer_idx: block_table.slot_mapping.gpu[:num_scheduled_tokens]
            for layer_idx, block_table in enumerate(input_batch.block_table.block_tables)
        }
        return PagedAttentionMetadata(
            kv_cache_tensor=kv_cache_tensor,
            block_tables=input_batch.block_table,
            slot_mapping=slot_mapping,
            query_start_loc=self.query_start_loc[: num_reqs + 1],
            seq_lens=self.seq_lens[:num_reqs],
            context_lens=self.context_lens[:num_reqs],
            query_lens=self.query_lens[:num_reqs],
            flat_positions=self.positions[:num_scheduled_tokens].to(dtype=torch.int32),
            token_request_indices=self.token_request_indices[:num_scheduled_tokens],
            num_reqs=num_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
        )

    def _run_forward(
        self,
        attn_metadata: PagedAttentionMetadata,
        num_scheduled_tokens: int,
    ) -> torch.Tensor:
        model = self._require_model()
        assert self.input_ids is not None
        assert self.positions is not None
        with set_forward_context(ForwardContext(attn_metadata=attn_metadata)):
            return model(
                input_ids=self.input_ids[:num_scheduled_tokens],
                positions=self.positions[:num_scheduled_tokens],
            )

    def _compute_logits(
        self,
        hidden_states: torch.Tensor,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        if logits_indices.numel() == 0:
            return hidden_states.new_empty((0, 0))
        sampled_hidden_states = hidden_states.index_select(0, logits_indices)
        return self._require_model().compute_logits(sampled_hidden_states)

    def _zero_new_blocks(self, block_ids: tuple[int, ...]) -> int:
        if not block_ids:
            return 0
        kv_cache_tensor = self._require_kv_cache_tensor()
        block_id_tensor = torch.tensor(
            block_ids,
            dtype=torch.long,
            device=kv_cache_tensor.device,
        )
        kv_cache_tensor.index_fill_(1, block_id_tensor, 0)
        logger.debug("Zeroed KV cache blocks count=%d block_ids=%s", len(block_ids), block_ids)
        return len(block_ids)

    @staticmethod
    def _maybe_extend_cached_request(
        request_state: CachedRequestState,
        new_token_ids: tuple[int, ...],
        num_computed_tokens: int,
    ) -> None:
        if num_computed_tokens >= request_state.num_tokens and new_token_ids:
            assert request_state.token_ids is not None
            request_state.token_ids.extend(new_token_ids)

    def _estimate_num_gpu_blocks(
        self,
        *,
        layer_specs: tuple[KVLayerSpec, ...],
        block_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
    ) -> int:
        first_spec = layer_specs[0]
        dtype = first_spec.dtype if isinstance(first_spec.dtype, torch.dtype) else torch.float16
        bytes_per_block = self._get_bytes_per_block(first_spec, dtype)
        device = self._resolve_device()

        if device.type == "cuda":
            free_memory, _total_memory = torch.cuda.mem_get_info(device)
            budget = int(free_memory * gpu_memory_utilization)
            return max(1, budget // bytes_per_block)

        blocks_per_layer = (max_model_len + block_size - 1) // block_size
        return max(1, len(layer_specs) * blocks_per_layer + 1)

    def _build_kv_cache_profile_result(
        self,
        *,
        layer_specs: tuple[KVLayerSpec, ...],
        block_size: int,
        num_gpu_blocks: int,
        gpu_memory_utilization: float,
    ) -> KVCacheProfileResult:
        first_spec = layer_specs[0]
        dtype = first_spec.dtype if isinstance(first_spec.dtype, torch.dtype) else torch.float16
        bytes_per_block = self._get_bytes_per_block(first_spec, dtype)
        device = self._resolve_device()
        available_memory_bytes: int | None = None
        memory_budget_bytes: int | None = None
        if device.type == "cuda":
            free_memory, _total_memory = torch.cuda.mem_get_info(device)
            available_memory_bytes = int(free_memory)
            memory_budget_bytes = int(free_memory * gpu_memory_utilization)

        usable_blocks = max(num_gpu_blocks - 1, 0)
        max_tokens_per_sequence = (usable_blocks // len(layer_specs)) * block_size
        return KVCacheProfileResult(
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
            bytes_per_block=bytes_per_block,
            available_memory_bytes=available_memory_bytes,
            memory_budget_bytes=memory_budget_bytes,
            max_tokens_per_sequence=max_tokens_per_sequence,
        )

    @staticmethod
    def _get_bytes_per_block(layer_spec: KVLayerSpec, dtype: torch.dtype) -> int:
        return (
            2
            * layer_spec.block_size
            * layer_spec.num_kv_heads
            * layer_spec.head_size
            * torch.empty((), dtype=dtype).element_size()
        )

    def _make_input_batch(self, kv_manager_config: KVManagerConfig) -> InputBatch:
        device = self._resolve_device()
        block_sizes = [spec.block_size for spec in kv_manager_config.layer_specs]
        return InputBatch(
            max_num_reqs=max(1, self.kvcore_config.scheduler_config.max_num_seqs),
            max_model_len=kv_manager_config.max_model_len,
            max_num_batched_tokens=max(
                1,
                self.kvcore_config.scheduler_config.max_num_scheduled_tokens,
            ),
            device=device,
            pin_memory=False,
            block_sizes=block_sizes,
            kernel_block_sizes=block_sizes,
        )

    def _init_step_buffers(self) -> None:
        input_batch = self._require_input_batch()
        device = self._resolve_device()
        self.input_ids = torch.empty(
            input_batch.max_num_batched_tokens,
            dtype=torch.long,
            device=device,
        )
        self.positions = torch.empty(
            input_batch.max_num_batched_tokens,
            dtype=torch.long,
            device=device,
        )
        self.query_start_loc = torch.empty(
            input_batch.max_num_reqs + 1,
            dtype=torch.int32,
            device=device,
        )
        self.seq_lens = torch.empty(input_batch.max_num_reqs, dtype=torch.int32, device=device)
        self.context_lens = torch.empty(
            input_batch.max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.query_lens = torch.empty(
            input_batch.max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.token_request_indices = torch.empty(
            input_batch.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def _require_step_buffers(self) -> None:
        if self.input_ids is None:
            raise RuntimeError("ModelRunner step buffers are not initialized")

    def _require_model(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call load_model first.")
        return self.model

    def _require_kv_cache_tensor(self) -> torch.Tensor:
        if self.kv_cache_tensor is None:
            raise RuntimeError(
                "KV cache tensor is not initialized. Call initialize_kv_cache first."
            )
        return self.kv_cache_tensor

    def _require_input_batch(self) -> InputBatch:
        if self.input_batch is None:
            raise RuntimeError(
                "InputBatch is not initialized. Call initialize_kv_cache first."
            )
        return self.input_batch

    def _resolve_device(self) -> torch.device:
        if self.kvcore_config.device_config.device is not None:
            return torch.device(self.kvcore_config.device_config.device)
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
        return torch.device("cpu")


__all__ = [
    "ExecuteModelState",
    "InputBatch",
    "KVCacheProfileResult",
    "ModelRunner",
    "ModelRunnerOutput",
    "ModelRunnerStepStats",
]
