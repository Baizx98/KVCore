from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from transformers import PretrainedConfig

from kvcore.kv.block_table import MultiGroupBlockTable
from kvcore.kv.kv_manager import KVManager, KVManagerConfig
from kvcore.model.kv_runtime import (
    KVForwardMetadata,
    SparseComputePlan,
)
from kvcore.model.model_loader import DefaultModelLoader, ModelLoadConfig


class ModelRunner:
    """Owns model creation and weight loading for the current process.

    This intentionally only covers the vLLM-like early lifecycle:
    create model skeleton -> load weights.
    Runtime concerns such as profiling, KV cache allocation, and execution
    loops are deferred to later work.
    """

    def __init__(self, load_config: ModelLoadConfig) -> None:
        self.load_config = load_config
        self.model_loader = DefaultModelLoader(load_config)
        self.hf_config: PretrainedConfig | None = None
        self.model: nn.Module | None = None
        self.kv_manager_config: KVManagerConfig | None = None
        self.kv_cache_tensor: torch.Tensor | None = None

    def create_model(self) -> nn.Module:
        self.hf_config = self.model_loader.load_config_from_source()
        self.model = self.model_loader.create_model(self.hf_config)
        return self.model

    def load_model(self) -> nn.Module:
        self.model = self.model_loader.load_model()
        self.hf_config = getattr(self.model, "config", None)
        return self.model

    def initialize_kv_cache(self, kv_manager_config: KVManagerConfig) -> torch.Tensor:
        """Create one global physical KV cache tensor.

        The scheduler owns the logical KVManager. ModelRunner only owns the
        runtime tensor and metadata needed to execute the model.
        """

        self.kv_manager_config = kv_manager_config
        self.kv_cache_tensor = self.initialize_kv_cache_tensor(kv_manager_config)
        return self.kv_cache_tensor

    def initialize_kv_cache_tensor(
        self,
        kv_manager_config: KVManagerConfig,
    ) -> torch.Tensor:
        """Allocate one contiguous KV cache tensor shared by all layers.

        `num_gpu_blocks` is the global physical block count across all layers.
        A layer finds its blocks only through block ids in attention metadata.
        """

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

    def build_block_tables(
        self,
        kv_manager: KVManager,
        request_ids: Sequence[str],
        sparse_plan: SparseComputePlan | None = None,
    ) -> MultiGroupBlockTable:
        max_num_batched_tokens = self._estimate_max_num_batched_tokens(
            kv_manager,
            request_ids,
            sparse_plan,
        )
        block_tables = MultiGroupBlockTable(
            max_num_reqs=max(1, len(request_ids)),
            max_model_len=kv_manager.config.max_model_len,
            max_num_batched_tokens=max(1, max_num_batched_tokens),
            pin_memory=False,
            device=self._resolve_device(),
            block_sizes=[spec.block_size for spec in kv_manager.config.layer_specs],
            kernel_block_sizes=[spec.block_size for spec in kv_manager.config.layer_specs],
        )
        self._last_logical_block_indices: dict[int, tuple[tuple[int, ...], ...]] = {}
        self._last_skipped_block_indices: dict[int, tuple[tuple[int, ...], ...]] = {}

        per_layer_rows: dict[int, list[list[int]]] = {
            layer_spec.layer_idx: [] for layer_spec in kv_manager.config.layer_specs
        }
        per_layer_logical: dict[int, list[tuple[int, ...]]] = {
            layer_spec.layer_idx: [] for layer_spec in kv_manager.config.layer_specs
        }
        per_layer_skipped: dict[int, list[tuple[int, ...]]] = {
            layer_spec.layer_idx: [] for layer_spec in kv_manager.config.layer_specs
        }

        for request_id in request_ids:
            kv_blocks = kv_manager.get_blocks(request_id)
            row_block_ids: list[list[int]] = []
            for layer_spec in kv_manager.config.layer_specs:
                blocks = kv_blocks.blocks[layer_spec.layer_idx]
                sparse_indices = (
                    set()
                    if sparse_plan is None
                    else sparse_plan.get_skip_indices(request_id, layer_spec.layer_idx)
                )
                block_ids: list[int] = []
                logical_indices: list[int] = []
                skipped_indices: list[int] = []
                for logical_index, block in enumerate(blocks):
                    if block.is_null or logical_index in sparse_indices:
                        skipped_indices.append(logical_index)
                        continue
                    block_ids.append(block.block_id)
                    logical_indices.append(logical_index)
                row_block_ids.append(block_ids)
                per_layer_rows[layer_spec.layer_idx].append(block_ids)
                per_layer_logical[layer_spec.layer_idx].append(tuple(logical_indices))
                per_layer_skipped[layer_spec.layer_idx].append(tuple(skipped_indices))
            block_tables.add_row(tuple(row_block_ids), len(per_layer_rows[0]) - 1)

        for layer_spec in kv_manager.config.layer_specs:
            layer_idx = layer_spec.layer_idx
            self._last_logical_block_indices[layer_idx] = tuple(per_layer_logical[layer_idx])
            self._last_skipped_block_indices[layer_idx] = tuple(per_layer_skipped[layer_idx])

        block_tables.commit_block_table(len(request_ids))
        return block_tables

    def build_slot_mapping(
        self,
        kv_manager: KVManager,
        block_tables: MultiGroupBlockTable,
    ) -> dict[int, torch.Tensor]:
        slot_mappings: dict[int, torch.Tensor] = {}
        for layer_idx, block_table in enumerate(block_tables.block_tables):
            query_start_loc, positions = self._build_query_positions_for_layer(
                kv_manager,
                layer_idx,
            )
            block_table.compute_slot_mapping(
                len(query_start_loc) - 1,
                query_start_loc,
                positions,
            )
            slot_mappings[layer_idx] = block_table.slot_mapping.gpu[: positions.numel()]
        return slot_mappings

    def build_attention_metadata(
        self,
        kv_manager: KVManager,
        request_ids: Sequence[str],
        sparse_plan: SparseComputePlan | None = None,
    ) -> KVForwardMetadata:
        kv_cache_tensor = self._require_kv_cache_tensor()
        block_tables = self.build_block_tables(
            kv_manager,
            request_ids,
            sparse_plan=sparse_plan,
        )
        return KVForwardMetadata(
            block_tables=block_tables,
            slot_mapping=self.build_slot_mapping(kv_manager, block_tables),
            kv_cache_tensor=kv_cache_tensor,
            sparse_plan=sparse_plan,
            logical_block_indices=getattr(self, "_last_logical_block_indices", None),
            skipped_block_indices=getattr(self, "_last_skipped_block_indices", None),
        )

    def _estimate_max_num_batched_tokens(
        self,
        kv_manager: KVManager,
        request_ids: Sequence[str],
        sparse_plan: SparseComputePlan | None,
    ) -> int:
        max_tokens = 0
        for request_id in request_ids:
            kv_blocks = kv_manager.get_blocks(request_id)
            for layer_spec in kv_manager.config.layer_specs:
                sparse_indices = (
                    set()
                    if sparse_plan is None
                    else sparse_plan.get_skip_indices(request_id, layer_spec.layer_idx)
                )
                active_blocks = sum(
                    not block.is_null and idx not in sparse_indices
                    for idx, block in enumerate(kv_blocks.blocks[layer_spec.layer_idx])
                )
                max_tokens = max(max_tokens, active_blocks * layer_spec.block_size)
        return max_tokens * max(1, len(request_ids))

    def _build_query_positions_for_layer(
        self,
        kv_manager: KVManager,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logical_by_layer = getattr(self, "_last_logical_block_indices", {})
        if not logical_by_layer or layer_idx not in logical_by_layer:
            return (
                torch.tensor([0], dtype=torch.int64, device=self._resolve_device()),
                torch.empty(0, dtype=torch.int64, device=self._resolve_device()),
            )

        block_size = kv_manager.config.layer_specs[layer_idx].block_size
        query_start_loc = [0]
        positions: list[int] = []
        for logical_blocks in logical_by_layer[layer_idx]:
            for logical_block in logical_blocks:
                start = logical_block * block_size
                positions.extend(range(start, start + block_size))
            query_start_loc.append(len(positions))
        device = self._resolve_device()
        return (
            torch.tensor(query_start_loc, dtype=torch.int64, device=device),
            torch.tensor(positions, dtype=torch.int64, device=device),
        )

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
    "ModelRunner",
]
