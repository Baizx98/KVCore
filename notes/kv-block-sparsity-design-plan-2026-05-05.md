# KVCore 块粒度 KV 稀疏设计初稿

## Summary

- 目标：在 KVCore 的现有推理基础设施上加入块粒度 KV cache 稀疏能力。
- 当前边界：`Scheduler/KVManager` 已维护完整逻辑块，`SchedulerOutput` 把 `block_ids` 传给 `ModelRunner`，`InputBatch` 再写入 runner-side `BlockTable`。
- 设计原则：参考 vLLM 哲学，`KVManager` 管逻辑 KV 生命周期和策略状态，`ModelRunner/InputBatch/BlockTable` 只管本 step 实际模型执行需要的 runtime metadata。
- 参考算法：`BlockWisePress` 的核心是块摘要、块内代表 key、尾部 query 窗口、head 聚合、保护 sink/recent/tail blocks。
- 当前阶段目标：先把设计问题列全，并给出一版理想实现路径；后续由用户在本文档中统一回答决策问题后再进入代码实现。

## 当前代码边界

- `Scheduler`：负责请求调度、KV 逻辑块分配、`SchedulerOutput` 构造。
- `KVManager`：负责逻辑块列表、物理块池、prefix cache、永久驱逐接口雏形。
- `ModelRunner`：负责模型加载、runner-side `InputBatch`、attention metadata 构造、forward 和 sampling。
- `InputBatch`：持久维护 request row、token ids、computed-token counts、runner-side block table。
- `BlockTable`：vLLM-style CPU/GPU staged block table，目前 attention kernel 按 `query_position // block_size` 直接索引逻辑块表。

## Key Changes

### 1. 配置层

新增 `SparseKVConfig` 并挂到 `KVCoreConfig`，默认关闭。

建议字段：

- `mode`: `disabled | permanent | dynamic | hybrid`
- `selection_interval`: `step | block | n_tokens`
- `selection_interval_tokens`: 当 interval 为 `n_tokens` 时使用
- `compression_ratio` 或 `keep_ratio`
- `q_window_size`
- `prefix_sink_blocks`
- `protected_recent_blocks`
- `score_ema_alpha`
- `summary_mode`
- `query_agg_mode`
- `head_agg_mode`
- `enable_prefill_sparsity`
- `enable_decode_sparsity`

推荐默认：

- `mode = disabled`
- 启用后优先实现 `dynamic`
- 稀疏先只作用于 decode
- dynamic 每 step 选择
- permanent 每新增完整 block 或每 `block_size` tokens 再执行

### 2. KVManager 逻辑状态

在 `KVManager` 内维护每个 `request_id/layer_idx/logical_block_idx` 的完整块状态。

建议状态字段：

- `request_id`
- `layer_idx`
- `logical_block_idx`
- `physical_block_id`
- `is_permanently_evicted`
- `was_dynamic_skipped`
- `score`
- `ema_score`
- `last_scored_step`
- `last_selected_step`
- `num_tokens`

设计目标：

- 永久驱逐改变未来的完整逻辑块状态。
- dynamic compute sparsity 只改变当前 forward 的可见 read block plan。
- prefix cache 跳过 permanent evicted blocks。
- 释放物理块时保留 tombstone 状态，避免逻辑 block index 漂移。

### 3. SchedulerOutput 契约

扩展 `SchedulerOutput`，保留完整逻辑块信息，同时新增 sparse read plan。

建议概念：

- `full_block_ids`: 完整逻辑块表，用于 KV 生命周期、slot write、debug。
- `visible_block_ids`: 本 step 实际参与 attention read 的块。
- `sparse_plan`: 描述每个 request/layer 被 keep 或 skip 的 logical block indices。
- `new_block_ids_to_zero`: 继续用于新物理块清零。

关键点：

- 不能简单把 skipped blocks 从当前 `BlockTable` 删除。
- 当前 attention backend 通过 `query_position // block_size` 直接索引 block table。 【是否可以通过将LogicBlock->PhysicalBlock的映射添加回来来保证逻辑位置和物理位置的正确匹配。因为我们的策略一定会保留尾端的块，那新kv写入的slot是一定在块表中的】
- 稀疏后必须区分 full write slot mapping 和 compact sparse read table。
- 否则会出现逻辑位置、物理写入位置、attention read 位置错位。

### 4. ModelRunner / InputBatch / BlockTable

`ModelRunner` 中维护两类 metadata：

- full metadata：用于当前新 token 的 KV write slot mapping。
- read metadata：用于 attention read，只包含本 step 可见块。

`InputBatch.block_table` 只应该包含本 step 实际参与 attention read 的 block table。

为了支持 compact read table，attention metadata 需要增加：

- compact read block table
- logical block index table 或 logical-to-compact mapping
- full slot mapping
- read token/position mapping

实现顺序建议：

1. 先在 `torch_paged` backend 中实现 sparse read correctness oracle。
2. 先保持 Triton dense-only，启用 sparse 时如果 backend 是 Triton 可先报错或 fallback 到 torch。
3. sparse metadata 稳定后再改 Triton read kernel。

### 5. BlockScoreCollector

在 `ModelRunner` 增加 `BlockScoreCollector`。

职责：

- 通过 `ForwardContext` 或 `Attention` wrapper 捕获每层 post-RoPE query。
- 维护每个请求最近 `q_window_size` 个 query。
- 基于 KV cache 中的 block summary 计算 block scores。
- 生成 `BlockScoreUpdate`，传回 `EngineCore/Scheduler/KVManager`。

建议 `BlockScoreUpdate` 字段：

- `request_id`
- `layer_idx`
- `logical_block_indices`
- `scores`
- `score_kind`
- `step_id`

推荐打分算法先移植 `BlockWisePress` 轻量版本：

- 块摘要：`mean_keys + topk_key_means`
- GQA：query heads 聚合到 KV heads
- query 聚合：默认 mean
- head 聚合：默认 uniform mean
- score 更新：默认 EMA
- 保护块：prefix sink、recent blocks、当前写入块、partial tail block

### 6. Feedback 链路

扩展 `ModelRunnerOutput`：

- 增加 `block_score_updates`

扩展 `EngineCore.step()`：

- `model_runner.execute_model()`
- `model_runner.sample_tokens()`
- 从 `ModelRunnerOutput` 取 sampled tokens 和 block scores
- 调用 `Scheduler.update_from_outputs(..., block_score_updates=...)`

扩展 `Scheduler.update_from_outputs()`：

- 正常推进 `num_computed_tokens`
- 调用 `KVManager.update_block_scores()`
- 由 `KVManager` 在下一次 `schedule()` 生成新的 sparse plan

## 需要用户决策的问题

### 1. 稀疏模式默认启用哪种？

推荐：`hybrid`，但先实现 dynamic compute sparsity，再接 permanent eviction。

选项：

- 只做 permanent
- 只做 dynamic
- hybrid

选择hybrid

### 2. 稀疏生效阶段？

推荐：decode 阶段先启用，prefill 默认 full attention；稳定后再扩到 chunked prefill。

选项：

- decode only
- prefill + decode
- prefill 后一次性压缩再 decode

选择decode only

### 3. 选择频率？

推荐：dynamic 每 step 选；permanent 每新增完整 block 或每 `block_size` tokens 执行。

选项：

- 每 step
- 每 `block_size` step
- 每 N tokens
- 只在 prefill 结束后

选择你推荐的

### 4. 块分数来源？

推荐：先用 query-aware summary score，不要求 backend 返回完整 attention weights。

选项：

- summary 近似分数
- 真实 attention weights 聚合
- 二者混合 EMA

选择summary 近似分数，不要求 backend 返回完整 attention weights，后续要支持flashattention，内核中kv的分数是单独计算的。

### 5. Q 窗口保存内容？

推荐：保存 post-RoPE query，按 KV head 聚合后的张量，放在 GPU runner state。

选项：

- pre-RoPE query
- post-RoPE query
- hidden_states 后算 q

选择保存 post-RoPE query

### 6. Permanent eviction 是否释放物理块？

推荐：释放物理块，并在逻辑状态中保留 tombstone，prefix cache 跳过被驱逐块。

选项：

- 释放物理块
- 保留物理块但永不读
- 只标记不释放，用于调试

选择释放物理块，像vllm中的slideattention那样使用nullblock代替呢

### 7. 被保护块规则？

推荐：保护 sink=1、recent=2、当前写入块、partial tail block。

选项：

- 固定保护
- 可配置保护
- 无保护作为 ablation

选择可配置保护，具体参数按照你推荐的来

### 8. Triton 支持节奏？

推荐：先让 `torch_paged` 成为 correctness oracle，再改 Triton read kernel 支持 compact sparse read table。

选项：

- torch first
- torch + triton 同步实现

选择torch first

### 9. Sparse plan 表达方式？

推荐：用 keep indices 作为主表达，skip indices 作为派生 debug 信息。

需要确认：

- `SchedulerOutput` 中传 keep 还是 skip？
- per-layer plan 是否允许不同层不同 keep set？
- 同一 request 的不同 batch item 是否需要同长度 compact read table？

- SchedulerOutput中都传，keep作为主表达
- 允许不同层不同keep set
- 同一 request 的不同 batch item 是否需要同长度 compact read table？这个我看不懂你的意思，你自己看着来吧

### 10. Score 生命周期？

推荐：per request/layer/block 保存 EMA score，request 结束时释放。

需要确认：

- score 是否需要跨请求复用？ 否
- prefix-cache 命中块是否继承历史 score？ 否
- permanent evicted block 是否保留最后一次 score 供分析？ 否

## 初步实现顺序

1. 文档和配置：新增 `SparseKVConfig`，默认关闭。
2. KVManager 状态：引入 block sparse state，不改变 dense 行为。
3. SchedulerOutput：增加 full/visible/sparse plan 字段，默认 visible 等于 full。
4. Runner metadata：拆分 full write slot mapping 和 sparse read table。
5. Torch attention：实现 sparse read oracle，保证 dense disabled 行为完全不变。
6. Score feedback：实现 `BlockScoreCollector` 和 `BlockScoreUpdate` 回传链路。
7. Dynamic policy：基于 EMA score 生成 decode-only visible plan。
8. Permanent policy：接入 tombstone + 释放物理块。
9. Triton：在 torch oracle 稳定后支持 sparse read kernel。

## Test Plan

### KVManager 单元测试

- sparse state 初始化和释放。
- permanent eviction 释放物理块并保留 tombstone。
- dynamic visible plan 不改变完整逻辑块表。
- prefix cache 跳过 permanent evicted blocks。

### Scheduler 测试

- `SparseKVConfig.mode=disabled` 时 `visible_block_ids == full_block_ids`。
- decode-only sparse 启用时，prefill 仍 full attention。
- protected sink/recent/current/tail blocks 不被跳过。
- `SchedulerOutput` 同时保留 full 和 visible 信息。

### ModelRunner 测试

- skipped blocks 不进入 read `InputBatch.block_table`。
- full slot mapping 仍能把新 token 写到正确物理 block。
- `ModelRunnerOutput.block_score_updates` 能按 request/layer/block 返回。

### Attention 正确性测试

- `torch_paged` sparse read 与 dense reference 对齐。
- disabled sparse 时现有 torch/triton paged attention 测试不变。
- sparse enabled + Triton 未支持时行为明确：fallback 或报错。

### Feedback 测试

- 构造固定 block scores。
- 验证 `ModelRunnerOutput -> EngineCore -> Scheduler.update_from_outputs -> KVManager.update_block_scores` 链路。
- 验证下一次 `schedule()` 使用更新后的 score 生成 sparse plan。

### E2E 测试

- 先跑 focused tests。
- 再用 `scripts/run_llm_engine_offline_batch.py` 做真实模型 smoke。
- 记录 sparse disabled baseline 和 sparse enabled 输出差异、latency、KV read blocks 数量。

## Assumptions

- 默认不引入 spec decode、LoRA、多模态、PP/DP、CUDA graph。
- 默认 Python-first；Triton kernel 等 sparse metadata 稳定后再做。
- 默认不为了历史接口保留兼容壳层。
- 默认先做 correctness，再做性能优化。
- 默认先支持 block-size aligned sparse read；partial tail block 保护不稀疏。
- 默认本设计里的用户决策问题由用户直接在本文档中补充答案。

## 2026-05-05 代码实现计划

### 决策归纳

- 稀疏模式：选择 `hybrid` 方向，但第一版先落地 dynamic compute sparsity 和 permanent eviction 状态基础。
- 生效阶段：decode only；prefill 保持 full attention。
- 选择频率：dynamic 每 step；permanent 后续按新增完整 block 或每 `block_size` tokens。
- 分数来源：summary 近似分数，不要求 backend 返回 attention weights。
- Q 窗口：保存 post-RoPE query，并按 KV head 聚合。
- Permanent eviction：释放物理块，并用 null block/tombstone 保持逻辑 index。
- 保护规则：可配置，默认 sink=1、recent=2、当前写入块、partial tail block。
- Triton：torch first；Triton sparse read 后续实现。
- Sparse plan 表达：`keep` 作为主表达，同时保留 `skip` debug；允许不同 layer 有不同 keep set。
- Score 生命周期：per request/layer/block，不跨请求复用；prefix-cache 命中不继承；evicted block 不保留分析 score。

### 实现步骤

1. 新增共享 sparse 数据结构和 `SparseKVConfig`，默认 `mode=disabled`，保证 dense 路径不变。
2. 在 `KVManager/SingleTypeKVManager` 中维护 per-request/layer/block sparse state、EMA score、tombstone 状态和 dynamic selection 状态。
3. 扩展 `SchedulerOutput/NewRequestData/CachedRequestData`，同时携带 full block ids、visible block ids、visible logical indices 和 `SparseKVPlan`。
4. 扩展 `InputBatch/BlockTable`：runner 侧保留 full block table 计算 write slot mapping，read block table 只放本 step 可见块，并用 logical block index side table 维持逻辑位置。
5. 修改 `torch_paged` backend：KV write 继续使用 full slot mapping；attention read 根据 compact read block table + logical indices 跳过不可见块。
6. 修改 `triton_paged` backend：当 read block table 不是 dense logical order 时明确报 `NotImplementedError`，避免静默错读。
7. 新增 `BlockScoreCollector`：通过通用 `Attention` wrapper 记录 post-RoPE query，forward 后基于 KV cache block summary 生成 `BlockScoreUpdate`。
8. 扩展 `ModelRunnerOutput -> EngineCore -> Scheduler.update_from_outputs -> KVManager.update_block_scores` feedback 链路。
9. 补充 focused tests：block table logical indices、scheduler decode-only sparse plan、torch sparse read oracle、runner block score feedback。

### 验证计划

- `uv run ruff check .`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_request_queue.py tests/test_scheduler.py tests/test_block_table.py tests/test_kv_manager.py tests/test_kv_compression.py tests/test_torch_paged_attention.py tests/test_model_runner_kv_boundary.py -q`
- `scripts/run_llm_engine_offline_batch.py` 做真实模型 E2E；模型加载阶段至少等待两分钟。
