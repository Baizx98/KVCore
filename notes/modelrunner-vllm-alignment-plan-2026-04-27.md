# ModelRunner 向 vLLM 对齐重构计划（包含 Chunked Prefill）

Date: 2026-04-27

> Status: historical plan. The current runtime contract was later tightened by
> the 2026-04-29 `InputBatch` alignment cleanup, which removed scheduler-owned
> flat input compatibility fields.

## Summary

目标：把 KVCore 的 `ModelRunner` 重构成单进程版 vLLM GPUModelRunner：同时承担 executor/worker 职责，维护 runner-side batch state，消费 scheduler 的增量输出，执行 prefill/decode/chunked prefill，并返回采样结果给 `Scheduler` 更新请求状态。

重点变化：保留当前 `Scheduler` 拥有逻辑 KV 生命周期、`ModelRunner` 拥有物理 KV tensor/metadata/forward 的边界；新增真正的 chunked prefill 支持，使长 prompt 可以按 token budget 多步执行，并在最后一个 prefill chunk 后采样进入 decode。

## Key Changes

- 重构 `SchedulerOutput` 为 vLLM 风格增量输出：
  - 区分 `scheduled_new_reqs` 和 `scheduled_cached_reqs`。
  - 每个请求携带本步 `num_scheduled_tokens`、`num_computed_tokens`、`block_ids`、`is_prefill`、`should_sample`、`sample_index`。
  - `total_num_scheduled_tokens` 替代当前散落的 flat 长度判断。
  - 历史第一阶段曾保留 `flat_input_token_ids / flat_positions` 兼容字段；当前运行时契约已删除这些 scheduler-owned flat 字段。

- 明确 chunked prefill 语义：
  - 当 prompt 长度超过 `max_num_scheduled_tokens` 或当前 batch budget 时，`Scheduler` 只调度一个 prompt chunk。
  - chunk 中间步：`is_prefill=True`，`should_sample=False`，只推进 KV 写入和 `num_computed_tokens`。
  - 最后一个 prompt chunk：`is_prefill=True`，`should_sample=True`，从该 chunk 最后一个 token 的 hidden state 采样首个 decode token。
  - 采样后请求进入 running decode；后续每步 decode 调度 1 token。
  - 不引入 vLLM 的 prefix cache hit 跳 token 优化以外的新策略；先保证 chunked prefill correctness。

- 引入 runner-side `InputBatch`：
  - 保存 `request_id -> token_ids / sampling_params / block_ids / num_computed_tokens`。
  - 新请求第一次出现时加入 runner state。
  - chunked prefill 后续 chunk 和 decode 作为 cached request 更新。
  - finished request 从 runner state 删除，但 KV free 仍由 `Scheduler` 完成。

- 拆分 `ModelRunner.execute_model()`：
  - `_update_states(scheduler_output)`：同步新请求、cached 请求和 block ids。
  - `_prepare_runtime_input(scheduler_output)`：生成本步 flat `input_ids`、`positions`、`sample_indices`、request-token mapping。
  - `_build_attention_metadata(...)`：从 runner batch state 的 block ids 构造 `PagedAttentionMetadata`，不再依赖传入 `KVManager`。
  - `_run_forward(...)`：只负责 forward context 和模型调用。
  - `_compute_logits_and_sample(...)`：只负责采样位置、logits 和 sampler。

- 调整 `EngineCore.step()`：
  - 调用 `scheduler.schedule()` 得到包含 block ids 的 `SchedulerOutput`。
  - 调用 `model_runner.execute_model(scheduler_output)`，不再传 `kv_manager`。
  - 调用 `scheduler.update_from_outputs()` 推进 `num_computed_tokens`、追加 sampled token、处理 stop/length、释放 KV。

## Test Plan

- Scheduler chunked prefill：
  - 长 prompt 在 token budget 下被拆成多步 prefill。
  - 中间 prefill chunk 不采样。
  - 最后 prefill chunk 采样一次。
  - decode 阶段每步只调度 1 个新 token。

- ModelRunner batch state：
  - 新请求、后续 prefill chunk、decode token 都能正确更新 runner state。
  - finished request 被 runner 删除。
  - runner 不调用 `KVManager.free()`。

- Runtime metadata：
  - chunked prefill 每个 chunk 的 `positions` 从真实 context offset 开始。
  - block table 和 slot mapping 覆盖当前 chunk 写入位置。
  - 多请求混合 batch 中，prefill chunk 和 decode request 的 sample index 正确。

- E2E：
  - tiny model 覆盖长 prompt chunked prefill -> 首 token 采样 -> 多步 decode -> finished。
  - 保留现有 `torch_paged` / `triton_paged` correctness 测试。
  - CUDA live run 作为慢测，不作为第一阶段强制门禁。

## Assumptions

- 第一阶段实现 chunked prefill correctness，不做 CUDA graph、async copy、prefix-cache benchmark、spec decode。
- Chunked prefill 的调度粒度由现有 `max_num_scheduled_tokens` 和 `max_num_seqs` 控制。
- `Scheduler` 仍是逻辑 KV owner，`ModelRunner` 只持有物理 KV tensor 和执行侧缓存。
