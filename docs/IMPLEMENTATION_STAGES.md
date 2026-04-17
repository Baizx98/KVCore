# Implementation Stages

本文档按照新的正确分层来定义实现顺序。

顺序固定为：

1. `model`
2. `model_runner`
3. `kvmanager`
4. `scheduler`
5. `kvoffload`
6. `engine/api` 集成

原因很简单：

- 没有 `model`，`ModelRunner` 无法定义输入输出
- 没有 `ModelRunner`，`KVManager` 不知道执行期需要的 KV 视图形态
- 没有 `KVManager`，`Scheduler` 无法做正确的容量决策
- 没有 `Scheduler`，`KVOffload` 也没有接入点
- 最后才是 `Engine` 把这些模块串成系统

---

## Stage 0: 文档基线

目标：

- 固定模块边界
- 固定接口方向
- 固定系统执行流程

退出条件：

- `docs/ARCHITECTURE.md` 与本文件成为后续重构的唯一设计基线

---

## Stage 1: Model

目标：

- 实现手写 decoder-only 模型结构
- 将 `llama3.py`、`qwen3.py`、`mistral3.py` 拆成独立模型文件
- 将共享层实现收口到 `model/layers/`
- 支持从 Hugging Face checkpoint 加载权重

必须完成的内容：

- `BaseCausalLM` 抽象
- `ModelRegistry`
- 手写 `Embedding / Attention / MLP / RMSNorm / DecoderLayer / LMHead`
- `KVCacheSpec`
- `load_weights()` 路径
- `get_kv_cache_spec()` 接口

退出条件：

- 可以不依赖 HF 模型对象执行 forward
- 可以加载至少一个 Hugging Face Llama checkpoint 并做前向
- `HF model_type -> KVCore model class` 映射成立
- 模型层命名与权重映射有测试覆盖

---

## Stage 2: ModelRunner

目标：

- 将执行打包逻辑全部收口到 `model_runner/`
- 删除 `runtime/`
- 明确 `prepare_input_batch -> execute -> sample` 三段职责

必须完成的内容：

- `ModelInputBatch`
- `AttentionMetadata`
- `StepOutput`
- `SampleOutput`
- `load_model()`
- 显式逐层执行路径

退出条件：

- `ModelRunner` 可以驱动手写 `model` 完成一步 prefill
- `ModelRunner` 可以驱动手写 `model` 完成一步 decode
- `ModelRunner` 成为 `Engine -> model` 的唯一执行适配入口
- 执行上下文和 layer runner 全部位于 `model_runner/`

---

## Stage 3: KVManager

目标：

- 以 `BlockPool` 为核心重做 KV 生命周期管理
- 删除独立 `allocator`
- 建立 request -> blocks -> layer views -> block tables 的完整逻辑链

必须完成的内容：

- `BlockPool`
- `SingleTypeKVManager`
- `KVManager`
- `RequestKVView`
- prefix cache 索引
- `BlockTable` 视图
- `ModelKVCacheSpec -> KVManager.initialize()` 路径

退出条件：

- `KVManager.allocate_slots()` 可测试
- `KVManager.get_request_view()` 可驱动 `ModelRunner`
- `BlockPool` 独立承担 block 分配/释放/缓存职责
- prefix cache 与 block 生命周期逻辑可验证
- 模型层只消费 KV 视图，不直接拥有 KV 生命周期

---

## Stage 4: Scheduler

目标：

- 以 step 为单位组织调度
- 将 `RequestState` 与 `ScheduledBatch` 收口到一个 scheduler 状态模块
- 调度时与 `KVManager` 联动做 token/block 预算决策

必须完成的内容：

- `RequestState`
- `ScheduledBatch` 或 `StepPlan`
- `Scheduler.schedule()`
- `Scheduler.update_from_output()`

第一阶段可接受的简化：

- 只支持单机
- 只支持基础 prefill/decode
- 不做复杂抢占

退出条件：

- `Scheduler` 不再只是简单队列
- `Scheduler` 调度时会查询 `KVManager`
- request 生命周期和 batch 描述统一归属 scheduler

---

## Stage 5: KVOffload

目标：

- 引入独立 `kvoffload/` 模块
- 为 `KVManager` 提供 block 级冷热迁移接口

必须完成的内容：

- `KVOffloadManager`
- `OffloadPlan`
- load/store/commit 三段式接口

第一阶段可接受的简化：

- 先支持 GPU <-> CPU
- 先不做多层级存储
- 先不做复杂 policy

退出条件：

- `KVManager` 可以调用 `KVOffloadManager`
- block 迁移后状态一致
- offload/load 有基础单测

---

## Stage 6: Engine / API 集成

目标：

- 用 `Engine` 将五个模块串起来
- 对外提供稳定的 `api.LLMEngine`

必须完成的内容：

- `Engine.add_request()`
- `Engine.step()`
- `Engine.generate()`
- `api.LLMEngine`

退出条件：

- 整体路径跑通：`add_request -> schedule -> execute -> sample -> update`
- 至少一个真实 Hugging Face checkpoint 可以完整生成
- 文档、接口、目录结构一致

---

## Stage 7: 评测与优化

目标：

- 在主路径正确后再引入优化

后续方向：

- continuous batching 强化
- chunked prefill
- prefix reuse 强化
- paged attention backend
- KV offload policy
- selective KV participation
- pruning

原则：

- 优化不能反向污染模块边界
- 所有优化都必须挂在既有接口上，而不是重新发明一套控制流

---

## 总结

后续实现必须严格遵守下面的顺序：

1. 先把 `model` 做对
2. 再把 `model_runner` 做对
3. 再把 `kvmanager` 做对
4. 再把 `scheduler` 做对
5. 再引入 `kvoffload`
6. 最后用 `engine/api` 完成系统闭环

如果顺序反了，最后一定会回到“接口全是临时补丁、模块职责混乱、代码大量冗余”的状态。

补充原则：

1. 新模型接入优先先做 `ModelRegistry + load_weights + KVCacheSpec`。
2. attention 适配优先服从 `attn_metadata + block_table + kv_cache_view` 接口。
3. 研究型改动如 KV compression / block importance / layer-wise offload，优先挂在
   `attention forward`、`KVManager`、`ModelRunner` 这三个 hook 点，而不是先改 `Engine`。
