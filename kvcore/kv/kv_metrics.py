from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass

from kvcore.kv.kv_utils import KVBlock


@dataclass(frozen=True, slots=True)
class KVCacheEvictionEvent:
    lifetime_seconds: float
    idle_seconds: float
    reuse_gaps_seconds: tuple[float, ...]


class BlockMetricsState:
    def __init__(self) -> None:
        now_ns = time.monotonic_ns()
        self.birth_time_ns = now_ns
        self.last_access_ns = now_ns
        self.access_history: deque[int] = deque(maxlen=4)

    def record_access(self) -> None:
        now_ns = time.monotonic_ns()
        self.last_access_ns = now_ns
        self.access_history.append(now_ns)

    def get_lifetime_seconds(self) -> float:
        return (time.monotonic_ns() - self.birth_time_ns) / 1e9

    def get_idle_time_seconds(self) -> float:
        return (time.monotonic_ns() - self.last_access_ns) / 1e9

    def get_reuse_gaps_seconds(self) -> list[float]:
        if len(self.access_history) < 2:
            return []
        history = list(self.access_history)
        return [(history[i] - history[i - 1]) / 1e9 for i in range(1, len(history))]


class KVCacheMetricsCollector:
    def __init__(self, sample_rate: float = 0.01) -> None:
        if not 0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in (0, 1.0], got {sample_rate}")
        self.sample_rate = sample_rate
        self.block_metrics: dict[int, BlockMetricsState] = {}
        self._eviction_events: list[KVCacheEvictionEvent] = []

    def should_sample_block(self) -> bool:
        return random.random() < self.sample_rate

    def on_block_allocated(self, block: KVBlock) -> None:
        if self.should_sample_block():
            self.block_metrics[block.block_id] = BlockMetricsState()

    def on_block_accessed(self, block: KVBlock) -> None:
        metrics = self.block_metrics.get(block.block_id)
        if metrics is not None:
            metrics.record_access()

    def on_block_evicted(self, block: KVBlock) -> None:
        metrics = self.block_metrics.pop(block.block_id, None)
        if metrics is None:
            return

        self._eviction_events.append(
            KVCacheEvictionEvent(
                lifetime_seconds=metrics.get_lifetime_seconds(),
                idle_seconds=metrics.get_idle_time_seconds(),
                reuse_gaps_seconds=tuple(metrics.get_reuse_gaps_seconds()),
            )
        )

    def reset(self) -> None:
        self.block_metrics.clear()
        self._eviction_events.clear()

    def drain_events(self) -> list[KVCacheEvictionEvent]:
        events = self._eviction_events
        self._eviction_events = []
        return events

