# ModelRunner 下一阶段重构计划

Date: 2026-04-27

> Status: superseded by the 2026-04-29 `InputBatch` alignment cleanup.
> `ModelRunnerBatch` has been renamed to `InputBatch`, and scheduler-owned
> `flat_input_token_ids / flat_positions` compatibility fields were removed
> from the runtime contract.

## 当前状态

已完成第一阶段 vLLM-style 对齐：

- `SchedulerOutput` 已区分 `scheduled_new_reqs` 和 `scheduled_cached_reqs`。
- `InputBatch` 已维护 runner-side request state。
- `EngineCore.step()` 已改为 `model_runner.execute_model(scheduler_output)`，`ModelRunner` 不再接收 `KVManager`。
- chunked prefill 已覆盖长 prompt 拆块、中间 chunk 不采样、最后 prefill chunk 采样、decode 每步 1 token。
- 真实 E2E 已跑通 `/Tan/model/Llama-3.1-8B-Instruct`，输入“中国的首都是什么？”，输出“中国的首都是北京。”。

当前边界仍保持：`Scheduler/KVManager` 管逻辑 KV 生命周期，`ModelRunner` 管物理 KV tensor、runner batch、metadata、forward、logits 和 sampling。

## 阶段 2 目标

目标不是补齐完整 vLLM serving stack，而是把当前最小闭环变成稳定的研究执行平台：

1. 让 `SchedulerOutput -> InputBatch -> PagedAttentionMetadata` 成为明确、可测试、可扩展的运行时契约。
2. 建立真实模型 correctness baseline，避免后续优化破坏 prefill/decode/chunked prefill 语义。
3. 给性能优化预留观测入口，先量化再优化。

## 计划 A：收敛 Scheduler/Runner 接口

### 问题

当前 `SchedulerOutput` 同时保留旧 flat 字段和新增量字段。这有利于迁移，但长期会造成两个事实来源：

- `flat_input_token_ids / flat_positions`
- `InputBatch` 根据 `scheduled_new_reqs / scheduled_cached_reqs` 重建的输入

### 假设

如果把 flat 输入完全下沉到 `ModelRunner._prepare_inputs()`，那么 scheduler 只需要表达调度决策，runner 负责执行输入构造，模块边界会更接近 vLLM。

### 方法

- 将 `flat_input_token_ids`、`flat_positions` 标记为兼容字段，只在测试中用于交叉校验。
- 增加 `SchedulerOutput.validate_consistency()` 或测试 helper，检查 flat 字段与增量字段等价。
- 新增 `ModelRunnerInputBuilder` 或私有方法组，集中构造：
  - input token ids
  - positions
  - query start loc
  - token request indices
  - sample indices
- `build_attention_metadata()` 只消费 `ModelRunnerInput` 的结构化中间结果，不直接再读 scheduler flat 字段。

### 最小验证

- 多请求混合 batch：一个 decode request + 一个 prefill chunk。
- 长 prompt chunked prefill：每个 chunk positions 连续且 sample index 正确。
- 对同一个 `SchedulerOutput`，旧 flat 字段与 runner 重建输入完全一致。

## 计划 B：建立真实模型 Correctness Baseline

### 问题

目前已有 tiny model 单测和一次真实 E2E，但还没有形成可复跑的真实模型 baseline。后续改 attention backend、metadata、scheduler 时，容易只看“能跑”，看不到输出语义是否漂移。

### 假设

用小模型 + 固定 prompt + greedy decoding，可以建立低成本 correctness guardrail；用 8B 模型作为慢测补充，不作为默认门禁。

### 方法

- 新增一个 repo-local live smoke script，例如 `scripts/run_live_generation.py`。
- 支持参数：
  - `--model /Tan/model/Llama-3.2-1B-Instruct`
  - `--device cuda:0`
  - `--prompt`
  - `--max-tokens`
  - `--num-gpu-blocks`
  - `--max-num-scheduled-tokens`
- 默认 prompt 使用中文短问答：“中国的首都是什么？”
- 输出固定记录：
  - model path
  - device
  - block size / num gpu blocks / max scheduled tokens
  - prompt
  - output text
  - output token ids
  - finish reason
  - engine init time / generation time
- 增加 pytest slow marker，只在本地模型存在且显式启用时运行。

### 最小验证

- Llama-3.2-1B-Instruct：默认 smoke，目标是低成本、可常跑。
- Llama-3.1-8B-Instruct：慢测，目标是和当前真实 E2E 证据对齐。
- 两个模型都使用 greedy decoding，避免采样随机性。

## 计划 C：加入 ModelRunner 执行观测

### 问题

下一阶段如果要做性能优化，当前只能看到最终生成时间，无法定位瓶颈是在 scheduler、metadata 构造、forward、logits 还是 sampling。

### 假设

先加入轻量 step-level metrics，可以指导后续是否优先优化 block table/slot mapping、Triton attention、KV cache 分配，或 batch compaction。

### 方法

- 增加 `ModelRunnerStepStats`，由 `execute_model()` 可选返回或挂到 runner last stats：
  - num requests
  - total scheduled tokens
  - num prefill reqs / num decode reqs
  - max query len / max seq len
  - prepare input time
  - metadata build time
  - forward time
  - logits/sample time
- 在 `EngineCore.step()` 层聚合最小 stats，不改变 `LLMEngine.generate()` 默认输出。
- 保持 metrics 纯 Python，不引入 tracing 框架。

### 最小验证

- tiny model 单测只检查 stats 字段存在且 token/request 数正确。
- live script 打印 per-run 汇总，默认不打印每步明细，避免输出过噪。

## 阶段 2 不做的事

- 不实现 vLLM 分布式 executor/worker。
- 不实现 CUDA graph。
- 不实现 spec decode。
- 不实现 CPU offload / KV transfer connector。
- 不把 scheduler 改成完整 priority/preemption serving scheduler。

这些都可以作为后续研究方向，但不应该阻塞当前 KVCore 形成稳定单机研究内核。

## 推荐执行顺序

1. 接口收敛：先让 runner 输入构造完全由 `InputBatch` 驱动，并补一致性测试。
2. Correctness baseline：加入 live generation script 和 slow smoke 测试。
3. 执行观测：加入 step stats，再基于真实数据选择优化点。
4. 文档同步：更新 `notes/current-architecture-flow.md`，把新 `SchedulerOutput / InputBatch` 契约写清楚。

## 成功标准

- `ModelRunner` 的核心执行路径不再依赖 scheduler flat 字段。
- chunked prefill、decode、多请求混合 batch 都有 targeted tests。
- 至少一个 1B 级本地模型 smoke 可以低成本复跑。
- 8B 级模型 E2E 有明确脚本和日志字段。
- 后续性能优化前，已经能看到 step-level 时间分布。
