from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from kvcore.config import KVCoreConfig, ModelConfig
from kvcore.kv.block_table import MultiGroupBlockTable
from kvcore.kv.kv_manager import KVManagerConfig
from kvcore.kv.single_type_kv_manager import KVLayerSpec
from kvcore.model.forward_context import ForwardContext, set_forward_context
from kvcore.model.kv_runtime import PagedAttentionMetadata
from kvcore.model.model_loader import DefaultModelLoader, ModelLoadConfig
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
class ModelStepOutput:
    sampled_request_ids: tuple[str, ...]
    sampled_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ModelRunnerPreparedInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    query_lens: torch.Tensor
    token_request_indices: torch.Tensor
    sample_indices: tuple[int, ...]
    sampled_request_ids: tuple[str, ...]
    sampling_params: tuple[SamplingParams, ...]
    num_reqs: int
    num_scheduled_tokens: int
    num_prefill_reqs: int
    num_decode_reqs: int
    max_query_len: int
    max_seq_len: int


@dataclass(frozen=True, slots=True)
class ModelRunnerInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    attn_metadata: PagedAttentionMetadata
    sample_indices: tuple[int, ...]
    sampled_request_ids: tuple[str, ...]
    sampling_params: tuple[SamplingParams, ...]


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


@dataclass(slots=True)
class ModelRunnerRequestState:
    req_id: str
    token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: tuple[tuple[int, ...], ...]
    num_computed_tokens: int = 0


class InputBatch:
    """Minimal row-based runner state, mirroring vLLM's InputBatch shape."""

    def __init__(self) -> None:
        self.req_ids: list[str] = []
        self.req_id_to_index: dict[str, int] = {}
        self.requests: list[ModelRunnerRequestState | None] = []

    def update_from_scheduler_output(self, scheduler_output: SchedulerOutput) -> None:
        for new_request in scheduler_output.scheduled_new_reqs:
            self.add_request(
                ModelRunnerRequestState(
                    req_id=new_request.req_id,
                    token_ids=list(new_request.prompt_token_ids),
                    sampling_params=new_request.sampling_params,
                    block_ids=new_request.block_ids,
                    num_computed_tokens=new_request.num_computed_tokens,
                )
            )
            logger.debug(
                "InputBatch added new request req_id=%s prompt_tokens=%d",
                new_request.req_id,
                len(new_request.prompt_token_ids),
            )
        cached_requests = scheduler_output.scheduled_cached_reqs
        for req_index, req_id in enumerate(cached_requests.req_ids):
            state = self.get_request(req_id)
            new_token_ids = cached_requests.new_token_ids[req_index]
            num_computed_tokens = cached_requests.num_computed_tokens[req_index]
            if num_computed_tokens >= len(state.token_ids):
                state.token_ids.extend(new_token_ids)
            state.block_ids = cached_requests.block_ids[req_index]
            state.num_computed_tokens = num_computed_tokens
            logger.debug(
                "InputBatch updated cached request req_id=%s new_tokens=%d "
                "num_computed_tokens=%d",
                req_id,
                len(new_token_ids),
                num_computed_tokens,
            )

    def add_request(self, request_state: ModelRunnerRequestState) -> None:
        if request_state.req_id in self.req_id_to_index:
            index = self.req_id_to_index[request_state.req_id]
            self.requests[index] = request_state
            return
        self.req_id_to_index[request_state.req_id] = len(self.req_ids)
        self.req_ids.append(request_state.req_id)
        self.requests.append(request_state)

    def remove_requests(self, req_ids: tuple[str, ...] | list[str]) -> None:
        removed = 0
        for req_id in req_ids:
            index = self.req_id_to_index.pop(req_id, None)
            if index is None:
                continue
            self.requests[index] = None
            removed += 1
        self._condense()
        if removed:
            logger.debug("InputBatch removed requests count=%d", removed)

    def _condense(self) -> None:
        active_requests = [request for request in self.requests if request is not None]
        self.req_ids = [request.req_id for request in active_requests]
        self.requests = active_requests
        self.req_id_to_index = {
            req_id: index for index, req_id in enumerate(self.req_ids)
        }

    def record_sampled_tokens(
        self,
        req_ids: tuple[str, ...],
        token_ids: tuple[int, ...],
    ) -> None:
        for req_id, token_id in zip(req_ids, token_ids, strict=True):
            state = self.get_request(req_id)
            state.token_ids.append(token_id)
        if req_ids:
            logger.debug("InputBatch recorded sampled tokens count=%d", len(req_ids))

    def get_request(self, req_id: str) -> ModelRunnerRequestState:
        try:
            index = self.req_id_to_index[req_id]
            request_state = self.requests[index]
        except KeyError as exc:
            raise KeyError(f"Request {req_id!r} is not present in InputBatch") from exc
        if request_state is None:
            raise KeyError(f"Request {req_id!r} is not present in InputBatch")
        return request_state


class ModelRunner:
    """Owns model creation, weight loading, and single-step execution."""

    def __init__(self, load_config: ModelLoadConfig | ModelConfig | KVCoreConfig) -> None:
        self.load_config = self._normalize_load_config(load_config)
        self.model_loader = DefaultModelLoader(self.load_config)
        self.hf_config: PretrainedConfig | None = None
        self.model: nn.Module | None = None
        self.kv_manager_config: KVManagerConfig | None = None
        self.kv_cache_tensor: torch.Tensor | None = None
        self.input_batch = InputBatch()
        self.sampler = Sampler()
        self.last_step_stats: ModelRunnerStepStats | None = None
        logger.info(
            "ModelRunner initialized model=%s device=%s attn_backend=%s",
            self.load_config.model,
            self.load_config.device,
            self.load_config.attn_backend,
        )

    def create_model(self) -> nn.Module:
        logger.info("Creating model model=%s", self.load_config.model)
        self.hf_config = self.model_loader.load_config_from_source()
        self.model = self.model_loader.create_model(self.hf_config)
        return self.model

    def load_model(self) -> nn.Module:
        logger.info("Loading model model=%s", self.load_config.model)
        self.model = self.model_loader.load_model()
        self.hf_config = getattr(self.model, "config", None)
        logger.info("Model loaded model=%s", self.load_config.model)
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
        """Resolve KV block capacity from runner-owned model/device state.

        This is the KVCore equivalent of vLLM keeping profiling in the model
        runner: the runner owns model/device memory and later owns the physical
        KV tensor, while the scheduler only receives the resolved block count.
        """
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

    def _prepare_runtime_input(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerPreparedInput:
        device = self._resolve_device()
        flat_input_token_ids: list[int] = []
        flat_positions: list[int] = []
        query_start_loc = [0]
        seq_lens: list[int] = []
        context_lens: list[int] = []
        query_lens: list[int] = []
        token_request_indices: list[int] = []
        sample_indices: list[int] = []
        sampled_request_ids: list[str] = []
        sampling_params: list[SamplingParams] = []
        for scheduled_request in scheduler_output.scheduled_requests:
            request_state = self.input_batch.get_request(scheduled_request.request_id)
            start = scheduled_request.num_computed_tokens
            end = start + scheduled_request.num_scheduled_tokens
            flat_input_token_ids.extend(request_state.token_ids[start:end])
            flat_positions.extend(range(start, end))
            request_idx = len(query_lens)
            query_start_loc.append(query_start_loc[-1] + scheduled_request.query_len)
            seq_lens.append(scheduled_request.context_len + scheduled_request.query_len)
            context_lens.append(scheduled_request.context_len)
            query_lens.append(scheduled_request.query_len)
            token_request_indices.extend([request_idx] * scheduled_request.query_len)
            if scheduled_request.should_sample:
                if scheduled_request.sample_index is None:
                    raise ValueError(
                        "Scheduled request marked for sampling must provide sample_index: "
                        f"{scheduled_request.request_id}"
                    )
                sample_indices.append(scheduled_request.sample_index)
                sampled_request_ids.append(scheduled_request.request_id)
                sampling_params.append(request_state.sampling_params)
        input_ids = torch.tensor(flat_input_token_ids, dtype=torch.long, device=device)
        positions = torch.tensor(flat_positions, dtype=torch.long, device=device)
        logger.debug(
            "Prepared model input reqs=%d tokens=%d samples=%d max_query_len=%d "
            "max_seq_len=%d",
            len(scheduler_output.scheduled_requests),
            scheduler_output.total_num_scheduled_tokens,
            len(sample_indices),
            max(query_lens, default=0),
            max(seq_lens, default=0),
        )
        return ModelRunnerPreparedInput(
            input_ids=input_ids,
            positions=positions,
            query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=device),
            query_lens=torch.tensor(query_lens, dtype=torch.int32, device=device),
            token_request_indices=torch.tensor(
                token_request_indices,
                dtype=torch.int32,
                device=device,
            ),
            sample_indices=tuple(sample_indices),
            sampled_request_ids=tuple(sampled_request_ids),
            sampling_params=tuple(sampling_params),
            num_reqs=len(scheduler_output.scheduled_requests),
            num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
            num_prefill_reqs=scheduler_output.num_prefill_reqs,
            num_decode_reqs=scheduler_output.num_decode_reqs,
            max_query_len=max(query_lens, default=0),
            max_seq_len=max(seq_lens, default=0),
        )

    def build_attention_metadata(
        self,
        scheduler_output: SchedulerOutput,
        prepared_input: ModelRunnerPreparedInput | None = None,
    ) -> PagedAttentionMetadata:
        prepared_input = prepared_input or self._prepare_runtime_input(scheduler_output)
        kv_cache_tensor = self._require_kv_cache_tensor()
        kv_manager_config = self._require_kv_manager_config()
        num_reqs = len(scheduler_output.scheduled_requests)
        device = self._resolve_device()

        block_tables = MultiGroupBlockTable(
            max_num_reqs=max(1, num_reqs),
            max_model_len=kv_manager_config.max_model_len,
            max_num_batched_tokens=max(1, scheduler_output.total_num_scheduled_tokens),
            pin_memory=False,
            device=device,
            block_sizes=[spec.block_size for spec in kv_manager_config.layer_specs],
            kernel_block_sizes=[spec.block_size for spec in kv_manager_config.layer_specs],
        )

        for request_idx, scheduled_request in enumerate(scheduler_output.scheduled_requests):
            block_ids = self.input_batch.get_request(scheduled_request.request_id).block_ids
            block_tables.add_row(block_ids, request_idx)

        block_tables.commit_block_table(num_reqs)

        slot_mapping: dict[int, torch.Tensor] = {}
        for layer_idx, block_table in enumerate(block_tables.block_tables):
            block_table.compute_slot_mapping(
                num_reqs,
                prepared_input.query_start_loc,
                prepared_input.positions.to(dtype=torch.int32),
            )
            slot_mapping[layer_idx] = block_table.slot_mapping.gpu[
                : scheduler_output.total_num_scheduled_tokens
            ]

        logger.debug(
            "Built attention metadata reqs=%d tokens=%d layers=%d",
            num_reqs,
            scheduler_output.total_num_scheduled_tokens,
            len(block_tables.block_tables),
        )
        return PagedAttentionMetadata(
            kv_cache_tensor=kv_cache_tensor,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            query_start_loc=prepared_input.query_start_loc,
            seq_lens=prepared_input.seq_lens,
            context_lens=prepared_input.context_lens,
            query_lens=prepared_input.query_lens,
            flat_positions=prepared_input.positions.to(dtype=torch.int32),
            token_request_indices=prepared_input.token_request_indices,
            num_reqs=num_reqs,
            num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
            num_prefill_reqs=scheduler_output.num_prefill_reqs,
            num_decode_reqs=scheduler_output.num_decode_reqs,
            max_query_len=prepared_input.max_query_len,
            max_seq_len=prepared_input.max_seq_len,
        )

    def prepare_model_input(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerInput:
        prepared_input = self._prepare_runtime_input(scheduler_output)
        attn_metadata = self.build_attention_metadata(scheduler_output, prepared_input)
        return self._to_model_runner_input(prepared_input, attn_metadata)

    @staticmethod
    def _to_model_runner_input(
        prepared_input: ModelRunnerPreparedInput,
        attn_metadata: PagedAttentionMetadata,
    ) -> ModelRunnerInput:
        return ModelRunnerInput(
            input_ids=prepared_input.input_ids,
            positions=prepared_input.positions,
            attn_metadata=attn_metadata,
            sample_indices=prepared_input.sample_indices,
            sampled_request_ids=prepared_input.sampled_request_ids,
            sampling_params=prepared_input.sampling_params,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelStepOutput:
        self.last_step_stats = None
        self._update_states(scheduler_output)
        if scheduler_output.is_empty:
            self.last_step_stats = ModelRunnerStepStats(0, 0, 0, 0, 0, 0, 0, 0, 0)
            logger.debug("ModelRunner skipped empty scheduler output")
            return ModelStepOutput((), ())
        if scheduler_output.total_num_scheduled_tokens == 0:
            self.last_step_stats = ModelRunnerStepStats(0, 0, 0, 0, 0, 0, 0, 0, 0)
            logger.debug("ModelRunner skipped zero-token scheduler output")
            return ModelStepOutput((), ())

        num_zeroed_blocks = self._zero_new_blocks(scheduler_output.new_block_ids_to_zero)
        prepare_start = perf_counter()
        prepared_input = self._prepare_runtime_input(scheduler_output)
        prepare_time = perf_counter() - prepare_start
        metadata_start = perf_counter()
        attn_metadata = self.build_attention_metadata(scheduler_output, prepared_input)
        metadata_time = perf_counter() - metadata_start
        model_input = self._to_model_runner_input(prepared_input, attn_metadata)
        forward_start = perf_counter()
        hidden_states = self._run_forward(model_input)
        forward_time = perf_counter() - forward_start
        sample_start = perf_counter()
        output = self._compute_logits_and_sample(hidden_states, model_input)
        sample_time = perf_counter() - sample_start
        self.last_step_stats = ModelRunnerStepStats(
            num_reqs=prepared_input.num_reqs,
            num_scheduled_tokens=prepared_input.num_scheduled_tokens,
            num_prefill_reqs=prepared_input.num_prefill_reqs,
            num_decode_reqs=prepared_input.num_decode_reqs,
            prepare_time_sec=prepare_time,
            metadata_time_sec=metadata_time,
            forward_time_sec=forward_time,
            sample_time_sec=sample_time,
            num_zeroed_blocks=num_zeroed_blocks,
        )
        logger.info(
            "ModelRunner step reqs=%d tokens=%d prefill=%d decode=%d "
            "samples=%d zeroed_blocks=%d prepare=%.6fs metadata=%.6fs "
            "forward=%.6fs sample=%.6fs",
            prepared_input.num_reqs,
            prepared_input.num_scheduled_tokens,
            prepared_input.num_prefill_reqs,
            prepared_input.num_decode_reqs,
            len(output.sampled_request_ids),
            num_zeroed_blocks,
            prepare_time,
            metadata_time,
            forward_time,
            sample_time,
        )
        return output

    def remove_requests(self, request_ids: tuple[str, ...] | list[str]) -> None:
        self.input_batch.remove_requests(request_ids)

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        self.input_batch.update_from_scheduler_output(scheduler_output)

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

    def _run_forward(self, model_input: ModelRunnerInput) -> torch.Tensor:
        model = self._require_model()
        with set_forward_context(ForwardContext(attn_metadata=model_input.attn_metadata)):
            return model(
                input_ids=model_input.input_ids,
                positions=model_input.positions,
            )

    def _compute_logits_and_sample(
        self,
        hidden_states: torch.Tensor,
        model_input: ModelRunnerInput,
    ) -> ModelStepOutput:
        if not model_input.sample_indices:
            logger.debug("Sampling skipped: no sample indices")
            return ModelStepOutput((), ())

        sample_index_tensor = torch.tensor(
            model_input.sample_indices,
            dtype=torch.long,
            device=hidden_states.device,
        )
        sampled_hidden_states = hidden_states.index_select(0, sample_index_tensor)
        model = self._require_model()
        logits = model.compute_logits(sampled_hidden_states)
        sampled_token_ids = self.sampler.sample(logits, model_input.sampling_params)
        sampled_token_ids_tuple = tuple(int(token_id) for token_id in sampled_token_ids.tolist())
        self.input_batch.record_sampled_tokens(
            model_input.sampled_request_ids,
            sampled_token_ids_tuple,
        )
        logger.debug(
            "Sampled tokens request_ids=%s token_ids=%s",
            model_input.sampled_request_ids,
            sampled_token_ids_tuple,
        )
        return ModelStepOutput(
            sampled_request_ids=model_input.sampled_request_ids,
            sampled_token_ids=sampled_token_ids_tuple,
        )

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

    def _require_model(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call create_model/load_model first.")
        return self.model

    def _require_kv_cache_tensor(self) -> torch.Tensor:
        if self.kv_cache_tensor is None:
            raise RuntimeError(
                "KV cache tensor is not initialized. Call initialize_kv_cache first."
            )
        return self.kv_cache_tensor

    def _require_kv_manager_config(self) -> KVManagerConfig:
        if self.kv_manager_config is None:
            raise RuntimeError(
                "KV manager config is not initialized. Call initialize_kv_cache first."
            )
        return self.kv_manager_config

    def _resolve_device(self) -> torch.device:
        if self.load_config.device is not None:
            return torch.device(self.load_config.device)
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
        return torch.device("cpu")

    @staticmethod
    def _normalize_load_config(
        load_config: ModelLoadConfig | ModelConfig | KVCoreConfig,
    ) -> ModelLoadConfig:
        if isinstance(load_config, KVCoreConfig):
            return load_config.model.to_load_config()
        if isinstance(load_config, ModelConfig):
            return load_config.to_load_config()
        return load_config


__all__ = [
    "KVCacheProfileResult",
    "InputBatch",
    "ModelRunner",
    "ModelRunnerInput",
    "ModelRunnerPreparedInput",
    "ModelRunnerRequestState",
    "ModelRunnerStepStats",
    "ModelStepOutput",
]
