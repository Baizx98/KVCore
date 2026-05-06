from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SparseKVMode(StrEnum):
    DISABLED = "disabled"
    PERMANENT = "permanent"
    DYNAMIC = "dynamic"


class SparseKVSelectionInterval(StrEnum):
    STEP = "step"
    BLOCK = "block"
    N_TOKENS = "n_tokens"


@dataclass(slots=True)
class BlockSparseState:
    request_id: str
    layer_idx: int
    logical_block_idx: int
    physical_block_id: int
    is_permanently_evicted: bool = False
    was_dynamic_skipped: bool = False
    score: float | None = None
    ema_score: float | None = None
    last_scored_step: int = -1
    last_selected_step: int = -1


@dataclass(frozen=True, slots=True)
class LayerSparsePlan:
    request_id: str
    layer_idx: int
    selected_block_indices: tuple[int, ...]
    evicted_block_indices: tuple[int, ...] = ()
    is_sparse: bool = True


@dataclass(frozen=True, slots=True)
class SparseKVPlan:
    layer_plans: tuple[LayerSparsePlan, ...] = ()

    @classmethod
    def empty(cls) -> SparseKVPlan:
        return cls(())

    @property
    def is_empty(self) -> bool:
        return not self.layer_plans

    def get_selected_indices(
        self,
        request_id: str,
        layer_idx: int,
    ) -> tuple[int, ...] | None:
        for plan in self.layer_plans:
            if plan.request_id == request_id and plan.layer_idx == layer_idx:
                return plan.selected_block_indices
        return None

    def get_evicted_indices(self, request_id: str, layer_idx: int) -> tuple[int, ...]:
        for plan in self.layer_plans:
            if plan.request_id == request_id and plan.layer_idx == layer_idx:
                return plan.evicted_block_indices
        return ()


@dataclass(frozen=True, slots=True)
class BlockScoreUpdate:
    request_id: str
    layer_idx: int
    logical_block_indices: tuple[int, ...]
    scores: tuple[float, ...]
    score_kind: str
    step_id: int

    def __post_init__(self) -> None:
        if len(self.logical_block_indices) != len(self.scores):
            raise ValueError(
                "logical_block_indices and scores must have the same length, "
                f"got {len(self.logical_block_indices)} and {len(self.scores)}"
            )


__all__ = [
    "BlockScoreUpdate",
    "BlockSparseState",
    "LayerSparsePlan",
    "SparseKVMode",
    "SparseKVPlan",
    "SparseKVSelectionInterval",
]
