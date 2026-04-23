from __future__ import annotations

import torch

from kvcore.kv.kv_manager import KVManagerConfig
from kvcore.kv.kv_metrics import KVCacheMetricsCollector
from kvcore.kv.single_type_kv_manager import KVLayerSpec
from kvcore.sched.scheduler import Scheduler


def make_kv_config() -> KVManagerConfig:
    return KVManagerConfig(
        num_gpu_blocks=8,
        max_model_len=16,
        layer_specs=(
            KVLayerSpec(
                layer_idx=0,
                block_size=2,
                num_kv_heads=2,
                head_size=8,
                dtype=torch.float16,
            ),
        ),
    )


def test_scheduler_initializes_and_owns_kv_manager() -> None:
    scheduler = Scheduler(make_kv_config())

    assert scheduler.kv_manager.config.num_gpu_blocks == 8
    assert scheduler.kv_manager.num_layers == 1


def test_scheduler_passes_metrics_collector_to_kv_manager() -> None:
    metrics_collector = KVCacheMetricsCollector(sample_rate=1.0)
    scheduler = Scheduler(make_kv_config(), metrics_collector=metrics_collector)

    assert scheduler.kv_manager.block_pool.metrics_collector is metrics_collector
