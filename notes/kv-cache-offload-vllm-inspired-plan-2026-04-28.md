# KV Cache CPU Offload 规划（参考 vLLM Connector）

Date: 2026-04-28

## 背景

本规划只讨论 **KV cache block-level CPU offload**，不实现代码。

需要明确区分两类概念：

- vLLM `--cpu-offload-gb` 主要是模型权重 offload，让 CPU 内存像“扩展 GPU 显存”一样参与权重常驻与搬运。
- vLLM 新 KV offloading connector 面向 KV cache 数据移动。`SimpleCPUOffloadConnector` 的文档说明它是 “CPU KV cache offloading with custom kernel transfers and BlockPool LRU”，并按 scheduler / worker role 拆分管理逻辑。
- vLLM 2026 KV offloading blog 说明，connector API 从同步读写发展到异步 loading/storing，目标是避免 KV 数据搬运阻塞新 batch，并提供可插拔 backend API。

参考：

- vLLM `SimpleCPUOffloadConnector`: https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector/
- vLLM KV offloading connector blog: https://vllm.ai/blog/kv-offloading-connector

## KVCore 设计目标

KVCore 不复制 vLLM 的 distributed connector 栈，而是抽出对研究有用的最小边界：

- `Scheduler/KVManager` 决定逻辑 block 生命周期和 offload/prefetch 计划。
- `ModelRunner` 执行真实 tensor copy 和 KV tensor 写回。
- CPU storage 作为 GPU KV blocks 的第二层存储。
- 第一阶段只做显式 offload/prefetch，不做自动 LRU、不做异步 DMA、不做 distributed transfer。

## 模块边界

### `KVBlockLocation`

表示逻辑 block 当前位置：

- `GPU`: block table 可直接引用 GPU KV tensor 中的物理 block id。
- `CPU`: KV 值在 CPU storage，进入 attention 前必须 prefetch。
- `NULL`: 已永久驱逐或逻辑上不可读。

### `KVOffloadManager`

归属：`kvcore/kv/`

职责：

- 维护 `request_id/layer_idx/logical_block_idx -> location`。
- 维护 CPU storage key 与 GPU physical block id 的映射。
- 生成 `KVTransferPlan`。
- 与 `KVManager` 协作释放或重新分配 GPU blocks。

不负责：

- torch tensor copy。
- attention metadata 构造。
- 自动调度策略。

### CPU KV Storage

第一版形态：

- 使用 pinned CPU tensor，布局与 GPU `kv_cache_tensor[:, block_id]` 单 block 切片一致。
- key 使用 `(request_id, layer_idx, logical_block_idx)`。
- 提供 `put_block()` / `get_block()` / `delete_block()`。

### `KVTransferPlan`

由 `KVOffloadManager` 生成，交给 `ModelRunner` 执行：

- `offload_blocks`: GPU -> CPU。
- `prefetch_blocks`: CPU -> GPU。
- `free_gpu_blocks`: copy 完成后释放。
- `required_before_forward`: 本 step attention 前必须完成的 prefetch。

### ModelRunner Tensor Copy Hook

`ModelRunner` 新增：

- `execute_kv_transfer_plan(plan)`。
- offload: 从 `kv_cache_tensor[:, gpu_block_id]` copy 到 CPU storage。
- prefetch: 从 CPU storage copy 回新 GPU block。
- stats: copy bytes、copy time、block count。

`build_attention_metadata()` 只能看到 GPU 或 NULL block；如果 scheduled request 中存在 CPU block 且没有 prefetch，直接报错。

## 第一阶段执行流程

### 显式 Offload

```text
user/research policy
  -> KVOffloadManager.plan_offload(request/layer/block_indices)
  -> ModelRunner.execute_kv_transfer_plan()
  -> KVManager marks logical blocks as CPU location and releases GPU blocks
```

### 显式 Prefetch

```text
Scheduler.schedule()
  -> sees scheduled request needs CPU blocks
  -> KVOffloadManager.plan_prefetch(...)
  -> allocate replacement GPU blocks
  -> ModelRunner.execute_kv_transfer_plan()
  -> SchedulerOutput block ids point to GPU-resident blocks
```

## 与当前 KVCore 模块的关系

- `BlockPool` 继续管理 GPU physical blocks。
- `KVManager` 继续管理 request/layer logical block table。
- `ModelRunner` 继续是唯一持有 GPU KV tensor 的模块。
- `PagedAttentionMetadata` 不直接表达 CPU block；进入 attention 前必须完成 prefetch。
- 永久驱逐和 CPU offload 是不同语义：
  - 永久驱逐：逻辑位置变为 NULL，后续读到空 block。
  - CPU offload：逻辑位置仍有效，但数据暂存在 CPU。

## 测试计划

- 单 block GPU -> CPU -> GPU roundtrip 后 tensor 完全一致。
- offload 后 GPU block 被释放，CPU storage 有对应 entry。
- prefetch 后 block table 指向新的 GPU block id。
- scheduled request 若包含 CPU block 且未 prefetch，metadata 构造报错。
- 多层、多 block offload/prefetch 后 `torch_paged` 输出与 no-offload 一致。

## 暂不实现

- 自动 LRU offload policy。
- 异步 copy / overlap。
- 自定义 transfer kernel。
- distributed KV connector。
- 权重 CPU offload。
- NVMe / disk storage。

## 后续研究方向

- 用 block hit rate 和 decode latency 驱动自动 offload。
- 分层策略：GPU hot blocks + CPU warm blocks + recompute/drop cold blocks。
- 与随机 KV sparse compression 结合：部分 block offload，部分 block 永久驱逐。
- 对比 vLLM native CPU KV offloading、LMCache、KV compression 方法的 throughput/latency tradeoff。
