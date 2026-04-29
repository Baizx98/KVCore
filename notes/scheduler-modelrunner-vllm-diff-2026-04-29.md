# Scheduler / ModelRunner 与 vLLM 最新实现差异对比

Date: 2026-04-29

## 对比范围

本文件对比当前 KVCore 工作树中的：

- `kvcore/sched/utils.py`
- `kvcore/sched/scheduler.py`
- `kvcore/model/model_runner.py`
- `kvcore/kv/compression.py`

与 vLLM main 分支中相近模块：

- `vllm/v1/core/sched/output.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu_input_batch.py`

参考源码：

- https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/sched/output.py
- https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/sched/scheduler.py
- https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu_model_runner.py
- https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu_input_batch.py

## 总体结论

KVCore 当前的方向已经和 vLLM V1 主线对齐到三个关键点：

1. `SchedulerOutput` 使用 new/cached request 增量协议。
2. `ModelRunner` 维护 runner-side request state，而不是每步从 scheduler 重发全量状态。
3. `ModelRunner` 从 scheduler output 构造 flattened token input、positions、block table、slot mapping 和 attention metadata。

但 KVCore 仍是研究内核级简化实现。主要差异是：

- vLLM scheduler 是通用 serving scheduler；KVCore scheduler 是单机 KV 研究调度器。
- vLLM `GPUModelRunner` 有高性能 persistent `InputBatch` 和大量预分配 CPU/GPU buffer；KVCore 现在也使用 `InputBatch` 命名和 row-based request index，但内部仍是 Python list state，没有 staged CPU/GPU tensor。
- vLLM 支持 multimodal、LoRA、pooling、spec decode、structured output、pipeline parallel、KV connector、encoder cache、async scheduling；KVCore 当前都没有或只保留规划。
- KVCore 额外加入了随机 KV block 永久驱逐接口，这是研究功能，不是 vLLM 同名路径。

## SchedulerOutput 差异

| 语义 | KVCore | vLLM 最新实现 | 差异判断 |
|---|---|---|---|
| new request | `NewRequestData` | `NewRequestData` | 命名已对齐。KVCore 只保留 `req_id`、`prompt_token_ids`、sampling params、block ids、computed/scheduled tokens；vLLM 还包含 mm features、pooling params、LoRA、prompt embeds、prefill token ids。 |
| cached request | `CachedRequestData` | `CachedRequestData` | 命名和批量容器形态已对齐。KVCore 保留 `req_ids/new_token_ids/block_ids/num_computed_tokens/num_scheduled_tokens` 等最小字段；vLLM 还包含 resumed reqs、all token ids、num output tokens 等。 |
| scheduled token count | `num_scheduled_tokens` dict + `total_num_scheduled_tokens` | `num_scheduled_tokens` dict + `total_num_scheduled_tokens` | 命名和语义对齐。 |
| finished notification | 无 scheduler-output 字段 | `finished_req_ids: set[str]` | KVCore 已删除未使用的 finished output 字段；finished 清理由 `SchedulerUpdateResult.finished_requests` 驱动 `EngineCore -> ModelRunner.remove_requests()`。 |
| new block zeroing | `new_block_ids_to_zero: tuple[int, ...]` | `new_block_ids_to_zero: list[int] | None` | 语义对齐，都是通知 worker/runner 清零新分配 block，避免 stale data/NaN。 |
| spec decode | 无 | `scheduled_spec_decode_tokens`、`num_invalid_spec_tokens` | KVCore 暂无 spec decode。 |
| encoder / multimodal | 无 | `scheduled_encoder_inputs`、`free_encoder_mm_hashes` | KVCore 暂无 encoder cache / multimodal。 |
| common prefix | 无 | `num_common_prefix_blocks` | KVCore 有 prefix cache，但未向 runner 暴露 cascade attention 所需 common prefix 信息。 |
| connectors | 无执行字段 | `kv_connector_metadata`、`ec_connector_metadata` | KVCore offload/connector 尚未执行。 |
| structured output | 无 | structured output flags | KVCore 暂无 grammar/structured output。 |

### 关键偏差

KVCore 的 `CachedRequestData.new_token_ids` 目前仍被 runner 用作通用增量 token 来源；vLLM 注释里 `new_token_ids` 主要服务 pipeline parallel，非 PP 情况下 runner 会更多依赖 persistent `InputBatch`、`prev_sampled_token_ids` 和 token buffer。KVCore 这样做是合理的研究简化，但后续如果实现 async scheduling/spec decode，需要重构。

## Scheduler 调度算法差异

| 维度 | KVCore | vLLM 最新实现 | 差异判断 |
|---|---|---|---|
| 调度核心 | 单一 catch-up 逻辑：让每个 request 的 `num_computed_tokens` 追上当前 `num_tokens` | 单一 catch-up 逻辑：让每个 request 的 `num_computed_tokens` 追上 `num_tokens_with_spec` | KVCore 已从 phase-based loop 改为 catch-up 模型；差异是暂不支持 spec token，所以目标长度是 `num_tokens`。 |
| request queue | `waiting: list[str]`、`running: list[str]` | `RequestQueue`，支持 policy / priority / skipped waiting | KVCore 缺少策略队列和 skipped waiting。 |
| partial prefill | `max_num_partial_prefills`、long prefill 限制 | scheduler config 中也有 partial prefill 相关限制，并融合在通用调度循环中 | 目标接近，但 KVCore 逻辑还很浅。 |
| KV manager | `KVManager`，每层 manager + shared block pool | `KVCacheManager`，支持 groups、events、DCP/PCP、Eagle、connector | KVCore 只保留研究核心。 |
| block zeroing | 每步 `take_new_block_ids()` 放入 output | vLLM 也在 output 中带 freshly allocated block ids | 对齐。 |
| prefix cache | `get_computed_blocks()` + full attention cache hit | 更完整，含 metrics/events/common prefix/cascade attention 支撑 | KVCore 还缺少 common prefix metadata。 |
| preemption | 无 | 有 preempted req ids / pause state / connector async loading state | KVCore 暂无。 |
| offload connector | 只有规划文档 | scheduler role connector，绑定 GPU block pool，处理 KV load/store metadata | KVCore 未接入执行路径。 |
| speculative decoding | 无 | Eagle/spec token/lookahead/output placeholders | KVCore 暂无。 |
| multimodal/encoder | 无 | encoder cache manager / multimodal budget | KVCore 暂无。 |

### 关键偏差

KVCore 当前已采用 per-request deficit / catch-up 模型，但排序仍简化为 running requests 优先、waiting requests 其次。vLLM 的 request queue 还有 priority/skipped waiting/preemption/connector 状态，因此仍更适合复杂 serving。

## InputBatch 差异

| 维度 | KVCore `InputBatch` | vLLM `InputBatch` / `CachedRequestState` | 差异判断 |
|---|---|---|---|
| 数据结构 | Python list rows + `req_id_to_index` + condense | 固定容量 batch rows + `req_id_to_index` + CPU/GPU staged buffers | KVCore 命名和 row mutation 边界已靠近 vLLM，但仍没有预分配 tensor buffers。 |
| token storage | 每请求 Python list | `token_ids_cpu_tensor[max_reqs, max_model_len]` + numpy view | KVCore 低开销开发但性能差；vLLM 可快速 index_select 和拷贝。 |
| block table | 每步 `MultiGroupBlockTable` 重新构造 | `InputBatch.block_table` 持久维护，add/remove/swap/condense | KVCore 目前每步构造成本更高。 |
| sampling params | 保存在 request state | 温度/top-p/top-k/penalty/logprobs 等都有 CPU/GPU tensor 和 request sets | KVCore Sampler 功能远少于 vLLM。 |
| batch compaction | `remove_requests()` 后 `_condense()` | `remove_request()` + `condense()` + `swap_states()` | KVCore 有最小 row compaction，但没有 swap states 和固定容量 buffer。 |
| sampled token cache | `record_sampled_tokens()` append 到 Python list | `prev_sampled_token_ids` GPU cache + optional async CPU copy | KVCore 简化同步路径；vLLM 优化 async / spec / PP。 |
| prompt embeds / multimodal | 无 | prompt embeds、mm features、M-RoPE/XD-RoPE metadata | KVCore 暂无。 |
| LoRA / pooling | 无 | LoRA mapping、pooling states | KVCore 暂无。 |

### 关键偏差

KVCore 新增 `ModelRunnerPreparedInput` 后，执行输入已从 runner-side state 生成，这是正确方向。但它仍然每步创建小 tensor 和 block table，不像 vLLM 那样维护持久 staged buffers。短期利于研究；一旦做性能评估，metadata 构造会成为明显瓶颈。

## ModelRunner 执行路径差异

| 维度 | KVCore `ModelRunner` | vLLM `GPUModelRunner` | 差异判断 |
|---|---|---|---|
| state update | `_update_states()` 更新 Python batch | `_update_states()` 更新 persistent `InputBatch`、spec/LoRA/mm/connector states | KVCore 只覆盖 text generation 基础路径。 |
| input prepare | `_prepare_runtime_input()` 构造 input_ids/positions/lens | `_prepare_inputs()` 使用 staged CPU/GPU buffers、block table commit overlap、positions/slot mapping GPU 化 | KVCore 语义对齐但性能简化。 |
| metadata | `PagedAttentionMetadata` 单一路径 | attention metadata builder 适配多 backend、CUDA graph、common metadata、spec decode | KVCore 简化。 |
| forward | 直接 `model(input_ids, positions)` | 支持 compilation/cudagraph/intermediate tensors/PP/DP/EP/routed experts | KVCore 暂无分布式和图优化。 |
| sample | `compute_logits` + `Sampler.sample` 返回 token ids | 完整 sampler/logprobs/prompt logprobs/async output copy/spec draft proposal | KVCore sampler 很小。 |
| stats | `ModelRunnerStepStats` 简单时间统计 | vLLM 有 `ModelMetrics`、`PerfStats`、CUDA graph stats、connector stats 等 | KVCore stats 是研究可观测入口，不等价生产 metrics。 |
| zeroing | `kv_cache_tensor.index_fill_(1, block_ids, 0)` | worker 根据 `new_block_ids_to_zero` 清理对应 GPU memory | 语义对齐，KVCore 当前未区分 KV group / non-attn cache。 |

## KV Sparse Compression 与 vLLM 差异

KVCore 的 `RandomKVBlockCompressor` 是新增研究接口：

- 随机选择 request/layer 下的非 null 逻辑 block。
- 默认跳过最后一个 block。
- 调用 `KVManager.evict_request_blocks()` 做永久驱逐。

vLLM 最新主线没有这个同名抽象。vLLM 更接近：

- 通过 KV cache manager / connector 管理 block 生命周期。
- 通过 offload / prefix cache / preemption / events 处理 KV 存储和转移。
- 不把“随机永久驱逐 block”作为标准 serving 功能。

所以该模块不是“对齐 vLLM”，而是 KVCore 面向 KV 稀疏压缩实验的扩展点。需要在文档和命名中明确它是 compression research hook，避免和 vLLM offload/preemption 混淆。

## 哪些变量/类命名接近但语义不同

| KVCore 名称 | vLLM 相近名称 | 差异 |
|---|---|---|
| `NewRequestData` | `NewRequestData` | 命名对齐；KVCore 字段少于 vLLM。 |
| `CachedRequestData` | `CachedRequestData` | 命名和批量形态对齐；KVCore 字段少于 vLLM。 |
| `num_scheduled_tokens` | `num_scheduled_tokens` | 命名和语义对齐。 |
| `InputBatch` | `InputBatch` | 命名和 row index 边界对齐；KVCore 未实现预分配 tensor buffer。 |
| `ModelRunnerPreparedInput` | `_prepare_inputs()` 中间张量集合 | KVCore 显式 dataclass；vLLM 使用 runner 内部预分配 buffers 和返回 logits/spec metadata。 |
| `ModelRunnerStepStats` | `PerfStats` / `ModelMetrics` | KVCore 是本地 step timing；vLLM 是更完整 observability 体系。 |
| `RandomKVBlockCompressor` | 无直接对应 | KVCore 研究扩展，不是 vLLM 标准调度功能。 |

## 对 KVCore 下一步建议

1. **短期保留当前简化结构**：现在最重要的是 correctness 和实验便利，不必马上复制 vLLM `InputBatch` 的复杂 tensor buffer。
2. **避免继续往 `SchedulerOutput` 放本地派生字段**：`is_prefill/should_sample/sample_index` 对 KVCore 方便，但 vLLM 更倾向 runner 基于 state 判断 discard/sample。后续 spec decode 前需要重新审视。
3. **性能前再引入 staged buffers**：如果 metadata/input 构造成为瓶颈，再把 `InputBatch` 的 token ids、positions、sampling params 改成预分配 CPU/GPU tensor。
4. **KV compression 与 offload 分开命名**：随机永久驱逐是 compression/sparsity hook；CPU offload 是 storage tiering hook，二者生命周期语义不同。
