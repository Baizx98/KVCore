from __future__ import annotations

from kvcore.kv.kv_manager import KVManager, KVManagerConfig
from kvcore.kv.kv_metrics import KVCacheMetricsCollector


class Scheduler:
    """Owns request scheduling state and logical KV block allocation.

    KVCore intentionally skips vLLM's executor/worker layers for now. That
    means the scheduler directly owns the KVManager, while ModelRunner owns
    physical KV cache tensors and model execution.
    """

    def __init__(
        self,
        kv_manager_config: KVManagerConfig,
        *,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.metrics_collector = metrics_collector
        self.kv_manager: KVManager = KVManager(
            kv_manager_config,
            metrics_collector=self.metrics_collector,
        )


__all__ = [
    "Scheduler",
]
