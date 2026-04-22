# KVBlock 与 BlockPool 最小实现设计记录

日期: 2026-04-21 00:55

## problem statement

当前 KV 目录下缺少可用的 KV block 元数据与 block pool 管理能力，后续无法支撑：

- KV block 分配
- KV block 释放
- prefix cache block 命中
- request block hash hook

目标是参考 vLLM 的 `KVCacheBlock / FreeKVCacheBlockQueue / BlockPool`，实现符合当前 KVCore 范围的最小版本。

## hypothesis

如果先实现以下最小能力：

- `KVBlock`
- `FreeKVBlockQueue`
- block hash helper
- `BlockPool`
- prefix cache hash -> block map

就可以支撑后续继续实现：

- block table
- KV manager
- scheduler 与 KV allocation 的交互

而不必提前引入 vLLM 中的事件系统、metrics、多模态、LoRA、复杂 hybrid KV cache。

## method

新增与修改：

- `kvcore/kv/kv_utils.py`
- `kvcore/kv/block_pool.py`
- `kvcore/kv/kv_metrics.py`
- `tests/test_kv_block_pool.py`

### `kv_utils.py`

实现：

- `BlockHash`
- `BlockHashWithGroupId`
- `make_block_hash_with_group_id`
- `get_block_hash`
- `get_group_id`
- `hash_block_tokens`
- `get_request_block_hasher`
- `KVBlock`
- `FreeKVBlockQueue`

### `block_pool.py`

实现：

- `BlockHashToBlockMap`
- `BlockPool`

### `kv_metrics.py`

实现：

- `KVCacheEvictionEvent`
- `BlockMetricsState`
- `KVCacheMetricsCollector`

当前 metrics 支持：

- 按 `sample_rate` 采样 block
- block allocated 时记录 birth time
- block accessed 时记录 last access 和最近 access history
- cached block evicted 时产出 eviction event
- reset 时清空 metrics 和 pending events

保留的核心语义：

- `block_id`
- `ref_cnt`
- block hash 只允许设置一次
- null block
- free queue
- `get_new_blocks`
- `free_blocks`
- `touch`
- `cache_full_blocks`
- `get_cached_block`
- `reset_prefix_cache`
- `get_usage`
- metrics collector hook

主动删减的复杂项：

- KV cache events
- LoRA / multimodal extra keys
- prompt embeds hash
- hybrid KV group block-size conversion
- distributed event reporting

## experiment

新增测试：

- `tests/test_kv_block_pool.py`

覆盖内容：

1. block hash + group id roundtrip
2. `KVBlock.block_hash` 只能设置一次
3. `FreeKVBlockQueue` pop/remove/append 顺序
4. `BlockPool` null block 预留
5. block 分配与释放
6. `touch` 从 free queue 中移除 block
7. full block prefix cache
8. cached block 被重新分配时驱逐
9. `reset_prefix_cache` 的成功/失败条件
10. metrics access / eviction event
11. metrics reset

执行结果：

- `ruff check`: 通过
- `pytest tests/test_kv_block_pool.py -q`: `11 passed in 0.05s`
- `compileall`: 通过

## conclusion

当前 `KVBlock / BlockPool` 已具备最小 KV block lifecycle 管理能力。

下一步可以继续实现：

- `BlockTable`
- `KVManager`
- request 到 block allocation 的接口
- slot mapping
