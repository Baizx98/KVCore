from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from kvcore.kv.block_table import MultiGroupBlockTable
from kvcore.kv.kv_manager import KVManager, KVManagerConfig
from kvcore.kv.single_type_kv_manager import KVLayerSpec
from kvcore.model.kv_runtime import PagedAttentionMetadata
from kvcore.model.model_loader import DefaultModelLoader, ModelLoadConfig
from kvcore.sample import Sampler
from kvcore.sched.utils import SchedulerOutput


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


class ModelRunner:
    """Owns model creation, weight loading, and single-step execution."""

    def __init__(self, load_config: ModelLoadConfig) -> None:
        self.load_config = load_config
        self.model_loader = DefaultModelLoader(load_config)
        self.hf_config: PretrainedConfig | None = None
        self.model: nn.Module | None = None
        self.kv_manager_config: KVManagerConfig | None = None
        self.kv_cache_tensor: torch.Tensor | None = None
        self.sampler = Sampler()

    def create_model(self) -> nn.Module:
        self.hf_config = self.model_loader.load_config_from_source()
        self.model = self.model_loader.create_model(self.hf_config)
        return self.model

    def load_model(self) -> nn.Module:
        self.model = self.model_loader.load_model()
        self.hf_config = getattr(self.model, "config", None)
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
        return self._build_kv_cache_profile_result(
            layer_specs=layer_specs,
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def initialize_kv_cache(self, kv_manager_config: KVManagerConfig) -> torch.Tensor:
        self.kv_manager_config = kv_manager_config
        self.kv_cache_tensor = self.initialize_kv_cache_tensor(kv_manager_config)
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

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self._resolve_device()
        input_ids = torch.tensor(
            scheduler_output.flat_input_token_ids,
            dtype=torch.long,
            device=device,
        )
        positions = torch.tensor(
            scheduler_output.flat_positions,
            dtype=torch.long,
            device=device,
        )
        return input_ids, positions

    def build_attention_metadata(
        self,
        kv_manager: KVManager,
        scheduler_output: SchedulerOutput,
    ) -> PagedAttentionMetadata:
        kv_cache_tensor = self._require_kv_cache_tensor()
        num_reqs = len(scheduler_output.scheduled_requests)
        device = self._resolve_device()

        block_tables = MultiGroupBlockTable(
            max_num_reqs=max(1, num_reqs),
            max_model_len=kv_manager.config.max_model_len,
            max_num_batched_tokens=max(1, scheduler_output.num_scheduled_tokens),
            pin_memory=False,
            device=device,
            block_sizes=[spec.block_size for spec in kv_manager.config.layer_specs],
            kernel_block_sizes=[spec.block_size for spec in kv_manager.config.layer_specs],
        )

        query_start_loc = [0]
        seq_lens: list[int] = []
        context_lens: list[int] = []
        query_lens: list[int] = []
        token_request_indices: list[int] = []

        for request_idx, scheduled_request in enumerate(scheduler_output.scheduled_requests):
            block_ids = kv_manager.get_block_ids(scheduled_request.request_id)
            block_tables.add_row(block_ids, request_idx)
            query_start_loc.append(query_start_loc[-1] + scheduled_request.query_len)
            seq_lens.append(scheduled_request.context_len + scheduled_request.query_len)
            context_lens.append(scheduled_request.context_len)
            query_lens.append(scheduled_request.query_len)
            token_request_indices.extend([request_idx] * scheduled_request.query_len)

        block_tables.commit_block_table(num_reqs)

        query_start_loc_tensor = torch.tensor(query_start_loc, dtype=torch.int32, device=device)
        flat_positions = torch.tensor(
            scheduler_output.flat_positions,
            dtype=torch.int32,
            device=device,
        )
        slot_mapping: dict[int, torch.Tensor] = {}
        for layer_idx, block_table in enumerate(block_tables.block_tables):
            block_table.compute_slot_mapping(
                num_reqs,
                query_start_loc_tensor,
                flat_positions,
            )
            slot_mapping[layer_idx] = block_table.slot_mapping.gpu[
                : scheduler_output.num_scheduled_tokens
            ]

        return PagedAttentionMetadata(
            kv_cache_tensor=kv_cache_tensor,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc_tensor,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=device),
            query_lens=torch.tensor(query_lens, dtype=torch.int32, device=device),
            flat_positions=flat_positions,
            token_request_indices=torch.tensor(
                token_request_indices,
                dtype=torch.int32,
                device=device,
            ),
            num_reqs=num_reqs,
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
            num_prefill_reqs=scheduler_output.num_prefill_reqs,
            num_decode_reqs=scheduler_output.num_decode_reqs,
            max_query_len=max(query_lens, default=0),
            max_seq_len=max(seq_lens, default=0),
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        kv_manager: KVManager,
    ) -> ModelStepOutput:
        if scheduler_output.is_empty:
            return ModelStepOutput((), ())

        model = self._require_model()
        input_ids, positions = self.prepare_inputs(scheduler_output)
        attn_metadata = self.build_attention_metadata(kv_manager, scheduler_output)
        hidden_states = model(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
        )

        sample_indices = [
            scheduled_request.sample_index
            for scheduled_request in scheduler_output.scheduled_requests
            if scheduled_request.should_sample and scheduled_request.sample_index is not None
        ]
        if not sample_indices:
            return ModelStepOutput((), ())

        sample_index_tensor = torch.tensor(
            sample_indices,
            dtype=torch.long,
            device=hidden_states.device,
        )
        sampled_hidden_states = hidden_states.index_select(0, sample_index_tensor)
        logits = model.compute_logits(sampled_hidden_states)
        sampled_token_ids = self.sampler.sample(logits, scheduler_output.sampling_params)
        return ModelStepOutput(
            sampled_request_ids=tuple(
                scheduled_request.request_id
                for scheduled_request in scheduler_output.scheduled_requests
                if scheduled_request.should_sample
            ),
            sampled_token_ids=tuple(int(token_id) for token_id in sampled_token_ids.tolist()),
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

    def _resolve_device(self) -> torch.device:
        if self.load_config.device is not None:
            return torch.device(self.load_config.device)
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
        return torch.device("cpu")


__all__ = [
    "KVCacheProfileResult",
    "ModelRunner",
    "ModelStepOutput",
]
