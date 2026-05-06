from __future__ import annotations

import math

import torch

from kvcore.config import SparseKVConfig
from kvcore.kv.sparse import BlockScoreUpdate
from kvcore.model.kv_runtime import PagedAttentionMetadata
from kvcore.sched.utils import SchedulerOutput
from kvcore.utils.log import get_logger

logger = get_logger(__name__)


class BlockScoreCollector:
    def __init__(self, config: SparseKVConfig) -> None:
        self.config = config
        self.step_id = 0
        self._queries_by_layer: dict[int, torch.Tensor] = {}
        self._query_windows: dict[tuple[str, int], torch.Tensor] = {}

    def begin_step(self) -> None:
        self._queries_by_layer.clear()

    def record_query(self, layer_idx: int, query: torch.Tensor) -> None:
        if not self.config.is_enabled:
            return
        self._queries_by_layer[layer_idx] = query.detach()

    def collect(
        self,
        *,
        req_ids: list[str],
        scheduler_output: SchedulerOutput,
        attn_metadata: PagedAttentionMetadata,
    ) -> tuple[BlockScoreUpdate, ...]:
        if not self.config.is_enabled:
            return ()
        updates: list[BlockScoreUpdate] = []
        key_cache = attn_metadata.kv_cache_tensor[0]
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        for layer_idx, query in self._queries_by_layer.items():
            kv_query = self._to_kv_query_heads(query, num_kv_heads)
            for req_row, req_id in enumerate(req_ids):
                query_indices = (
                    attn_metadata.token_request_indices == req_row
                ).nonzero(as_tuple=False).flatten()
                if query_indices.numel() == 0:
                    continue
                window = self._update_query_window(
                    req_id,
                    layer_idx,
                    kv_query.index_select(0, query_indices),
                )
                block_ids = self._get_full_block_ids(scheduler_output, req_id, layer_idx)
                if not block_ids:
                    continue
                block_indices, scores = self._score_blocks(
                    key_cache=key_cache,
                    q_window=window,
                    block_ids=block_ids,
                    block_size=block_size,
                )
                if block_indices:
                    updates.append(
                        BlockScoreUpdate(
                            request_id=req_id,
                            layer_idx=layer_idx,
                            logical_block_indices=block_indices,
                            scores=scores,
                            score_kind="summary_mean_topk_query",
                            step_id=scheduler_output.step_id,
                        )
                    )
        if updates:
            logger.info(
                "Sparse KV block scores collected updates=%d layers=%s requests=%s step_id=%d",
                len(updates),
                tuple(sorted({update.layer_idx for update in updates})),
                tuple(sorted({update.request_id for update in updates})),
                scheduler_output.step_id,
            )
        return tuple(updates)

    def clear_requests(self, request_ids: tuple[str, ...] | list[str]) -> None:
        request_id_set = set(request_ids)
        for key in tuple(self._query_windows):
            if key[0] in request_id_set:
                self._query_windows.pop(key, None)

    def _update_query_window(
        self,
        req_id: str,
        layer_idx: int,
        new_queries: torch.Tensor,
    ) -> torch.Tensor:
        key = (req_id, layer_idx)
        previous = self._query_windows.get(key)
        window = new_queries if previous is None else torch.cat([previous, new_queries], dim=0)
        if window.shape[0] > self.config.q_window_size:
            window = window[-self.config.q_window_size :]
        self._query_windows[key] = window.detach()
        return window

    def _score_blocks(
        self,
        *,
        key_cache: torch.Tensor,
        q_window: torch.Tensor,
        block_ids: tuple[int, ...],
        block_size: int,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        logical_indices: list[int] = []
        summaries: list[torch.Tensor] = []
        topk_summaries: list[torch.Tensor] = []
        topk = min(self.config.summary_topk_keys, block_size)
        for logical_idx, block_id in enumerate(block_ids):
            if block_id == 0:
                continue
            block_keys = key_cache[block_id]
            summaries.append(block_keys.mean(dim=0))
            selector_scores = block_keys.norm(dim=-1)
            topk_indices = selector_scores.topk(topk, dim=0).indices
            gather_index = topk_indices[..., None].expand(-1, -1, block_keys.shape[-1])
            topk_keys = block_keys.gather(0, gather_index)
            topk_summaries.append(topk_keys.mean(dim=0))
            logical_indices.append(logical_idx)
        if not summaries:
            return (), ()

        mean_keys = torch.stack(summaries, dim=0)
        topk_keys = torch.stack(topk_summaries, dim=0)
        mean_scores = self._score_summary(q_window, mean_keys)
        topk_scores = self._score_summary(q_window, topk_keys)
        scores = (
            self.config.mean_key_weight * mean_scores
            + (1.0 - self.config.mean_key_weight) * topk_scores
        )
        return tuple(logical_indices), tuple(float(score) for score in scores.tolist())

    @staticmethod
    def _score_summary(q_window: torch.Tensor, summary_keys: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum(
            "qhd,bhd->qhb",
            q_window.to(torch.float32),
            summary_keys.to(torch.float32),
        )
        scores = scores / math.sqrt(q_window.shape[-1])
        return scores.mean(dim=(0, 1))

    @staticmethod
    def _to_kv_query_heads(query: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
        num_query_heads = query.shape[1]
        if num_query_heads == num_kv_heads:
            return query
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_kv_heads, "
                f"got {num_query_heads} and {num_kv_heads}"
            )
        groups = num_query_heads // num_kv_heads
        return query.view(query.shape[0], num_kv_heads, groups, query.shape[2]).mean(dim=2)

    @staticmethod
    def _get_full_block_ids(
        scheduler_output: SchedulerOutput,
        req_id: str,
        layer_idx: int,
    ) -> tuple[int, ...]:
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id == req_id:
                return new_req.block_ids[layer_idx]
        cached = scheduler_output.scheduled_cached_reqs
        for req_index, cached_req_id in enumerate(cached.req_ids):
            if cached_req_id == req_id:
                return cached.block_ids[req_index][layer_idx]
        return ()


__all__ = [
    "BlockScoreCollector",
]
