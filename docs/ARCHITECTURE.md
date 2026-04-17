# Architecture

本文档重新定义 KVCore 的目标架构。

设计基线参考 vLLM 最新 V1 思路，但不会机械复制其完整 serving stack。
尤其是下面几条是本项目的明确约束：

- 不引入独立 `worker` 层，原本由 worker 承担的单机执行能力并入 `ModelRunner`
- 不保留单独 `runtime/` 包
- 不保留单独 `allocator`，逻辑/物理 block 分配统一由 `BlockPool` 负责
- `model` 必须是手写模型结构，负责模块定义与 Hugging Face 权重加载
- `ModelRunner` 负责执行控制，不负责定义模型结构
- `Scheduler` 保持简洁，但接口形态要兼容未来 continuous batching / chunked prefill / offload

---

## 1. 设计目标

KVCore 的核心目标不是“先跑通一个最短 demo”，而是构建一个适合系统研究的 LLM 推理内核。

因此架构必须满足：

- `model`、`model runner`、`kv manager`、`scheduler`、`kv offload` 五层职责清晰
- 推理控制平面与模型计算平面分离
- KV cache 是一等公民，而不是附属缓存
- 接口先为后续研究留余地，再考虑优化
- 单机版本先把主路径做对，再逐步引入更复杂的调度与 offload

---

## 2. 模块顺序

KVCore 应按下面的顺序理解和实现：

1. `model`
2. `model_runner`
3. `kvmanager`
4. `scheduler`
5. `kvoffload`
6. `engine` / `api`

这不是随意排列，而是依赖关系的方向：

- `model_runner` 依赖 `model`
- `kvmanager` 为 `model_runner` 和 `scheduler` 提供 KV 能力
- `scheduler` 依赖 `kvmanager` 的容量与块视图做决策
- `kvoffload` 为 `kvmanager` 提供冷热分层与迁移能力
- `engine` 只负责把这些模块串成一步一步的推理循环

---

## 3. Model

### 3.1 职责

`model` 层负责定义“真正的网络结构”。

这层不是 Hugging Face 模型的薄包装器，而应像 vLLM 的 `model_executor/models/*.py` 一样：

- 手写模型模块
- 手写 layer 结构
- 手写 attention / MLP / norm / embedding / output head 的组合关系
- 提供 Hugging Face 权重加载接口
- 提供模型族特定的 KV cache 规格描述

换句话说，`model` 负责回答：

- 这个模型有哪些层
- 每层怎么前向
- 每层需要怎样的 KV cache
- Hugging Face 的权重如何映射到本地模块

### 3.2 目录建议

```text
kvcore/model/
  __init__.py
  registry.py
  base.py
  llama3.py
  qwen3.py
  mistral3.py
  layers/
    attention.py
    decoder.py
    embedding.py
    mlp.py
    norm.py
    rotary.py
```

### 3.3 核心接口

建议的模型接口如下：

```python
class BaseCausalLM(nn.Module):
    @classmethod
    def from_hf_config(cls, hf_config, model_config) -> "BaseCausalLM": ...

    def load_weights(self, hf_state_dict: dict[str, Tensor]) -> None: ...

    def get_kv_cache_spec(self) -> "ModelKVCacheSpec": ...

    def embed_input_ids(self, input_ids: Tensor) -> Tensor: ...

    def iter_layers(self) -> tuple[nn.Module, ...]: ...

    def forward_layer(
        self,
        layer_id: int,
        hidden_states: Tensor,
        positions: Tensor,
        kv_cache_group: "LayerKVCacheView",
        attn_metadata: "AttentionMetadata",
    ) -> Tensor: ...

    def finalize_hidden_states(self, hidden_states: Tensor) -> Tensor: ...

    def compute_logits(self, hidden_states: Tensor) -> Tensor: ...
```

### 3.4 各接口作用

- `from_hf_config`
  - 根据 HF config 构建本地模型拓扑
  - 只做结构初始化，不加载权重

- `load_weights`
  - 把 Hugging Face state dict 映射到本地模块参数
  - 这一步必须由 KVCore 自己控制，而不是把 HF 模型对象直接拿来执行

- `get_kv_cache_spec`
  - 返回模型层的 KV cache 规格
  - 包括层数、head 维度、kv head 数、dtype、block size 约束等
  - 这是 `KVManager` 的输入之一

- `embed_input_ids`
  - 将 token ids 变成 hidden states
  - 由模型层负责，因为 embedding 是模型结构的一部分

- `iter_layers`
  - 以固定顺序暴露 decoder layers

- `forward_layer`
  - 执行单层前向
  - 由 `ModelRunner` 驱动，但真正的层计算在 `model` 内

- `finalize_hidden_states`
  - 执行最终 norm 等尾部逻辑

- `compute_logits`
  - 计算输出 logits

### 3.5 当前明确结论

- `model` 必须是“手写层实现 + HF 权重加载”
- `model` 不应该依赖 HF 模型对象来执行 forward
- `model` 内必须保留模型族独立文件，如 `llama3.py`、`qwen3.py`、`mistral3.py`

### 3.6 模型适配流程

参考 vLLM 最新模型接入流程，一个新模型接入 KVCore 时，至少要完成下面五步：

1. 模型注册
2. 模型结构实现
3. attention 适配
4. KV cache 规格定义
5. 权重加载

对应到 KVCore 中，建议接口如下：

```python
class ModelRegistry:
    def register(self, model_type: str, model_cls: type[BaseCausalLM]) -> None: ...

    def resolve(self, model_type: str) -> type[BaseCausalLM]: ...
```

```python
class BaseCausalLM(nn.Module):
    @classmethod
    def from_hf_config(cls, hf_config, model_config) -> "BaseCausalLM": ...

    def get_kv_cache_spec(self) -> "ModelKVCacheSpec": ...

    def load_weights(self, weights_iterator) -> None: ...
```

#### 第一步：模型注册

目的：

- 建立 `HF model_type -> KVCore model class` 映射
- 让 `Engine.from_pretrained()` 能根据 Hugging Face config 找到正确模型类

#### 第二步：模型结构实现

目的：

- 定义手写模型 forward 拓扑
- 定义逐层执行接口
- 让模型服从 KV 外部管理，而不是内部自行维护 cache

#### 第三步：Attention 适配

这是模型接入里最关键的一层。

attention 必须支持：

- prefill / decode 两种模式
- 从 `attn_metadata` 读取 block table / slot mapping / positions
- 通过外部 `kv_cache` 视图读写 KV

也就是说，模型层里的 attention 不是“自己拥有 KV”，而是“消费 KV 抽象”。

#### 第四步：KVCacheSpec

每个模型族必须提供自己的 KV cache 规格描述，例如：

- num layers
- num attention heads
- num kv heads
- head dim
- block size 约束
- 是否 sliding window
- 是否 cross attention

`KVManager` 和 `ModelRunner` 都依赖这个规格对象初始化自己的内部结构。

#### 第五步：权重加载

权重加载必须是显式接口，不应隐式依赖 Hugging Face 模型本体前向。

建议的加载接口是：

```python
def load_weights(self, weights_iterator) -> None: ...
```

原因：

- 便于后续做 tensor parallel shard
- 便于做参数名映射与 reshape / transpose
- 便于做模型族差异化处理

### 3.7 模型适配发生在哪

从引擎视角看，一次模型适配发生在下面这条链上：

```text
Engine.from_pretrained()
  -> ModelRegistry.resolve()
  -> ModelRunner.load_model()
  -> BaseCausalLM.from_hf_config()
  -> BaseCausalLM.load_weights()
  -> attention forward (with attn_metadata / kv_cache / block_table)
```

这里有一个重要原则：

- `Engine` 不关心具体是什么模型
- `Scheduler` 不关心模型层怎么读写 KV
- 只要模型实现服从 `KV cache spec + attention metadata + load_weights` 接口，就能接入系统

### 3.8 KV 抽象优先

KVCore 必须和 vLLM 一样，坚持下面这条原则：

- KV 不属于模型
- KV 属于 `KVManager`

模型只是 KV 的读写消费者。

因此模型层不能做这样的事情：

```python
self.k_cache[layer_id] = ...
```

而应只消费：

- `kv_cache_view`
- `block_table`
- `slot_mapping`
- `attn_metadata`

这条原则对于你后续要做的：

- block importance
- KV compression
- layer-wise offload
- dynamic KV selection

都是基础前提

---

## 4. ModelRunner

### 4.1 职责

`ModelRunner` 是执行控制器，不是模型结构定义器。

它负责把 scheduler 产出的“本轮执行计划”变成一次具体模型前向。

其职责应包括：

- 根据调度结果准备输入张量
- 构造 positions / attention metadata / block tables
- 绑定本轮所需的 KV cache 视图
- 驱动模型逐层执行
- 执行 logits 后处理与采样前整理
- 向 engine 返回一步执行结果

在 vLLM 中，这部分能力分散在 worker + model runner + model input builder 里。
在 KVCore 单机版本中，这些能力统一收口到 `ModelRunner`。

### 4.2 `ModelRunner` 不应做什么

- 不定义模型层结构
- 不拥有请求调度逻辑
- 不决定 block 是否 offload
- 不直接管理 prefix cache 策略

### 4.3 目录建议

```text
kvcore/model_runner/
  __init__.py
  runner.py
  input_batch.py
  metadata.py
  output.py
  sampling.py
```

### 4.4 核心接口

```python
class ModelRunner:
    def __init__(self, model: BaseCausalLM, kv_manager: KVManager): ...

    def load_model(self, model_config: EngineConfig) -> BaseCausalLM: ...

    def profile_run(self) -> "RunnerProfile": ...

    def initialize_kv_cache(self, kv_cache_config: "KVCacheConfig") -> None: ...

    def prepare_input_batch(self, step_plan: "StepPlan") -> "ModelInputBatch": ...

    def execute(self, step_plan: "StepPlan") -> "StepOutput": ...

    def sample(self, step_output: "StepOutput") -> "SampleOutput": ...
```

### 4.5 各接口作用

- `load_model`
  - 通过 `ModelRegistry` 找到模型类
  - 构建手写模型对象
  - 触发 `load_weights()`
  - 这是 `Engine` 和 `model` 真正对接的位置

- `profile_run`
  - 做容量估计和 profile
  - 为 `KVManager` 初始化提供模型侧信息

- `initialize_kv_cache`
  - 根据 KV spec 与配置初始化运行时 cache tensor
  - 这一步由 runner 持有具体张量，但不拥有逻辑块生命周期

- `prepare_input_batch`
  - 将 scheduler 的请求计划压缩成模型可执行输入
  - 包括 `input_ids / positions / attn_metadata / block_tables`

- `execute`
  - 一步执行 prefill 或 decode
  - 逐层调用 `model.forward_layer`
  - 读取/写入由 `KVManager` 提供的 KV 视图

- `sample`
  - 从 logits 得到下一个 token
  - 这一步虽然可以继续拆，但保留在 `ModelRunner` 最清楚

### 4.6 设计结论

- `ModelRunner` 吸收原本 worker 内的执行整理逻辑
- `ModelRunner` 是“执行打包层”，不是“模型定义层”
- `runtime/` 不应该存在，相关内容都应并入 `model_runner`
- `ModelRunner` 是模型适配落地的第一执行层，但不是模型结构定义层

---

## 5. KVManager

### 5.1 职责

`KVManager` 是 KV cache 的逻辑控制平面。

它不直接定义张量计算，但负责：

- request 到 block 的映射
- block 的分配、释放、复用
- prefix cache 命中查找
- 逻辑 block table 维护
- 为 `ModelRunner` 提供每层的 KV 视图
- 与 `KVOffload` 协同做冷热迁移

### 5.2 内部层次

单机版建议保留下面这几层：

```text
KVManager
  -> SingleTypeKVManager (每种 attention / 每类层)
  -> BlockPool
  -> PrefixCacheIndex
  -> BlockTableView
```

当前不引入：

- `KVCacheCoordinator`
- 多 worker connector
- 分布式 KV transfer

### 5.3 模型只消费 KV，不拥有 KV

在 KVCore 中，模型接入必须服从下面这条 KV 抽象：

- `KVManager` 负责逻辑块生命周期
- `BlockPool` 负责物理 block 生命周期
- `BlockTable` 负责将逻辑 block id 映射成执行期可消费视图
- `model` / `attention` 只负责消费这些视图

这意味着：

- 不能把 KV cache 直接塞进模型对象内部做生命周期管理
- 不能让模型决定 block eviction / reuse / offload
- attention 只应通过 `attn_metadata + kv_cache_view` 工作

### 5.4 `BlockPool` 的定位

在新的设计里，`BlockPool` 同时承担：

- 物理 block 生命周期管理
- free queue
- cached block hash 索引
- block id 分配

所以单独的 `BlockAllocator` 应彻底删除。

### 5.5 核心接口

```python
class KVManager:
    def initialize(self, model_kv_spec: "ModelKVCacheSpec") -> None: ...

    def get_computed_prefix(self, request: "RequestState") -> "PrefixHitResult": ...

    def allocate_slots(self, request: "RequestState", num_new_tokens: int) -> "AllocationResult": ...

    def get_request_view(self, request_id: str) -> "RequestKVView": ...

    def commit_step(self, request_id: str, num_committed_tokens: int) -> None: ...

    def cache_full_blocks(self, request_id: str, token_ids: list[int]) -> None: ...

    def free(self, request_id: str) -> None: ...

    def take_events(self) -> list["KVEvent"]: ...
```

### 5.6 各接口作用

- `initialize`
  - 根据模型的 KV spec 初始化 block pool 和 manager 内部结构

- `get_computed_prefix`
  - 查询 prefix cache 命中

- `allocate_slots`
  - 为当前 step 分配需要写入的新 block / slot
  - 同时检查 block budget 是否足够

- `get_request_view`
  - 返回某个 request 当前的逻辑 KV 视图
  - 这是 `ModelRunner` 构建 attention metadata 的核心输入

- `commit_step`
  - 在一次前向真正完成后提交 token 数和 block 状态
  - 把“计划分配”变成“已写入状态”

- `cache_full_blocks`
  - 将完整块加入 prefix cache

- `free`
  - 请求结束后释放 block
  - 默认应支持尾块优先进入可驱逐状态

- `take_events`
  - 导出调试或可观测事件

---

## 6. Scheduler

### 6.1 职责

`Scheduler` 是系统的请求状态机与步级调度器。

它负责：

- 接收新请求
- 管理 waiting / running / finished 状态
- 组织本轮 prefill / decode
- 依据 token budget / block budget 选择本轮执行集
- 与 `KVManager` 协同决定本轮能否推进请求

单机第一版可以保持简单，但接口方向必须正确。

### 6.2 状态文件设计

`RequestState` 与 `ScheduledBatch` 不应拆在两个极小文件里。
它们本质上都是 scheduler 私有的状态封装，应该收口在一个模块，例如：

```text
kvcore/scheduler/
  scheduler.py
  state.py
```

### 6.3 核心接口

```python
class Scheduler:
    def add_request(self, request: Request) -> None: ...

    def abort_request(self, request_id: str) -> None: ...

    def has_pending_requests(self) -> bool: ...

    def schedule(self, kv_manager: KVManager) -> "StepPlan | None": ...

    def update_from_output(
        self,
        step_plan: "StepPlan",
        sample_output: "SampleOutput",
        kv_manager: KVManager,
    ) -> "FinishedRequest | None": ...
```

### 6.4 `schedule` 的输出

`schedule` 不应该只返回“一个 batch”。
更准确的概念应该是 `StepPlan`，至少包含：

- 本轮 mode：`prefill` / `decode` / `mixed`
- 本轮选中的 requests
- token budget 使用
- block budget 使用
- 每个 request 的 planned token count
- 每个 request 的 KV allocation result
- 供 `ModelRunner` 准备张量的紧凑元数据

也就是说，调度输出必须同时携带：

- 请求视角的信息
- KV 视角的信息
- 执行视角的信息

### 6.5 设计结论

- `Scheduler` 可以简单，但不能只做 queue pop/push
- `Scheduler` 和 `KVManager` 在一步调度里必须协同
- 后续 continuous batching / chunked prefill / prefix reuse 都要从这个接口长出来
- `Scheduler` 不应感知模型内部实现细节，只应消费模型提供的 KV 规格与运行时预算

---

## 7. KVOffload

### 7.1 职责

`KVOffload` 是 KV 层次化存储与迁移模块。

它不直接做调度，但要为 `KVManager` 提供：

- 哪些 block 应写回 CPU / SSD
- 哪些 block 应预取回 GPU
- 迁移的实际执行接口
- 迁移完成后的状态提交

### 7.2 设计要求

- 作为独立模块存在
- 由 `KVManager` 调用，不由 `Scheduler` 直接操作底层数据
- 迁移粒度以 block 为主
- 后续可以扩展到 layer-aware policy

### 7.3 核心接口

```python
class KVOffloadManager:
    def plan_load(self, blocks: list[int]) -> "OffloadPlan": ...

    def plan_store(self, blocks: list[int]) -> "OffloadPlan": ...

    def execute_load(self, plan: "OffloadPlan") -> None: ...

    def execute_store(self, plan: "OffloadPlan") -> None: ...

    def commit(self, plan: "OffloadPlan") -> None: ...
```

### 7.4 与系统的关系

- `Scheduler` 只关心“本轮是否可运行”
- `KVManager` 负责决定“哪些 block 需要 offload/load”
- `KVOffload` 负责执行迁移
- `ModelRunner` 只消费已经就绪的 KV 视图

---

## 8. Engine / API

### 8.1 API 层

`api.LLMEngine` 是请求入口。

职责：

- 接收 `prompt / sampling params`
- 构造内部 request
- 调用内部 `engine.Engine`
- 返回最终结果

它不应拥有调度逻辑或 KV 逻辑。

### 8.2 Engine 层

`engine.Engine` 是内部协调器。

职责：

- 拥有 `ModelRunner / KVManager / Scheduler / KVOffload`
- 驱动整个 step loop
- 处理 schedule -> execute -> sample -> update 的闭环

核心接口建议：

```python
class Engine:
    @classmethod
    def from_pretrained(cls, config: EngineConfig) -> "Engine": ...

    def add_request(self, request: Request) -> None: ...

    def step(self) -> list[FinishedRequest]: ...

    def generate(self, request: Request, generation_config: GenerationConfig) -> GenerationResult: ...
```

---

## 9. 系统整体执行流程

在 KVCore 中，一次请求的正确执行顺序应为：

1. `api.LLMEngine` 接收 prompt 和 sampling params
2. `Engine.add_request()` 创建内部请求状态
3. `Scheduler.schedule()` 选择本轮要推进的 requests，并向 `KVManager` 请求 block/slot 规划
4. `KVManager` 返回当前 request 的 KV 视图、prefix hit、allocation result
5. `Engine` 将 `StepPlan` 交给 `ModelRunner`
6. `ModelRunner.prepare_input_batch()` 整理 `input_ids / positions / block tables / attn metadata`
7. `ModelRunner.execute()` 驱动手写 `model` 逐层前向，并写入 KV cache
8. `ModelRunner.sample()` 返回本轮 token 结果
9. `Scheduler.update_from_output()` 更新请求状态，判断 finished / running
10. `KVManager.commit_step()` / `cache_full_blocks()` / `free()` 更新块生命周期
11. 如有需要，`KVOffload` 执行 load/store
12. `Engine` 进入下一轮 step，直到请求结束

如果把模型适配路径单独展开，则是：

1. `Engine.from_pretrained()` 读取 Hugging Face config
2. `ModelRegistry` 根据 `model_type` 找到本地模型类
3. `ModelRunner.load_model()` 构建手写模型
4. 手写模型通过 `load_weights()` 加载 Hugging Face 权重
5. `ModelRunner` 在执行时构造 `attn_metadata / block_tables / kv_cache_view`
6. 模型层 attention 基于这些元数据读写 KV

这条主路径中最重要的三条依赖关系是：

- `Scheduler` 依赖 `KVManager` 做容量判断
- `ModelRunner` 依赖 `model` 做真正前向
- `KVOffload` 由 `KVManager` 驱动，而不是反过来主导系统

---

## 10. 目录目标

推荐的目标目录为：

```text
kvcore/
  api/
  engine/
  model/
    layers/
  model_runner/
  kv/
  scheduler/
  kvoffload/
docs/
tests/
example/
```

其中：

- `model/` 只放模型定义与权重加载
- `model_runner/` 只放执行控制
- `kv/` 只放 KV 生命周期管理
- `scheduler/` 只放请求状态机与一步调度
- `kvoffload/` 只放分层存储迁移

---

## 11. 当前结论

后续实现必须遵守下面这些结论：

1. `model` 是手写模型，不是 HF 模型对象包装器。
2. `model_runner` 是执行控制层，不是模型结构层。
3. `BlockPool` 取代独立 `allocator`。
4. `scheduler` 的状态描述收口到一个模块。
5. `kvoffload` 必须作为独立模块存在，即使第一阶段先留接口。
6. `engine` 只编排模块，不侵入各子模块内部策略。
