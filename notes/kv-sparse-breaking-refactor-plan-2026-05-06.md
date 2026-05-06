# KV Sparse Breaking Refactor Plan - 2026-05-06

## Problem Statement

目标不是实现 sliding-window attention，而是借鉴 vLLM 的 null block/tombstone 思想，重构 KV sparse：

- KV cache 压缩的本质：让部分 token 的 KV 对本次 attention 不可见。
- `Scheduler/KVManager` 维护完整 token ids、完整逻辑 KV block 表、下一次计算的 sparse plan。
- permanent eviction 释放物理块，但不能让物理释放影响 prefix cache、slot mapping、logical block index。
- dynamic compute sparsity 保留物理块，只改变下一次 attention 可见性。
- `ModelRunner` 消费 `SchedulerOutput`，构造模型输入、sparse attention metadata，并计算最新 block score。
- 单机单卡 NVIDIA GPU；不保留无用 fallback、兼容路径、多设备抽象。

## Key Decision

破坏式重构后，主路径保留一个统一语义：

```text
full logical KV table + SparseKVPlan -> ModelRunner materializes dense/sparse attention metadata -> Triton paged attention
```

不再保留：

- scheduler 直接输出 `visible_block_ids`
- scheduler 侧 dense/sparse 双路径兼容桥
- torch fallback 作为 runtime path
- sliding-window manager 实现
- `hybrid` 这种含义模糊的 mode

`torch_paged` 可以继续作为测试 oracle，但不是 runtime fallback。

注意：attention backend 需要同时支持 dense 和 sparse 两种运行模式，因为一个 batch 里可能只有部分请求被稀疏，也可能一个请求都不稀疏。这里的“不要兼容”指不保留旧 scheduler/runner 契约，不是说 backend 只能跑 sparse。

## Mask 性能判断

如果在 attention 层额外传一个 selected mask，让 kernel 在内部“掩码提取实际参与计算的 KV 块”，性能是否会损失，取决于 mask 怎么用。

### 会明显损失的方式

如果 kernel 仍然遍历完整 `0..query_position` / 完整 block table，然后对每个 block 读 `selected_mask` 判断是否跳过：

- 会多一次 global memory mask load。
- warp 内分支可能发散。
- 如果仍扫描所有 blocks，计算量没有按压缩比例下降，只是少做部分 dot-product。
- 对 decode 阶段，长上下文下 block loop 仍然是 O(total_blocks)，不是真正 O(selected_blocks)。
- 如果 mask 是 token-level dense mask，metadata 带宽和 mask check 代价更高。

这种方式不建议作为最终实现。

### 可以接受的方式

如果 `ModelRunner` 预先把 sparse plan 物化成 compact selected block list，kernel 只遍历 selected blocks：

```text
selected_block_ids: [physical_block_id...]
selected_logical_block_indices: [logical_block_idx...]
```

kernel loop 变成 O(selected_blocks)，mask 不参与内层筛选，只用于构造 compact table。这样才会带来真实速度收益。

因此推荐：不要传 dense selected mask 给 kernel 做过滤；在 `ModelRunner` 中事先把 sparse metadata 构建好，传 compact selected block table + logical block indices，让 kernel 直接只读可见 blocks。kernel 不实时判断哪些 block 该参与计算。

## Target Data Model

为了不引入太多额外数据结构，只保留 4 类 sparse 相关对象：

### 1. `BlockSparseState`

位置：`kvcore/kv/sparse.py`

职责：KVManager 侧 per request/layer/logical block 的状态。

字段建议：

```python
request_id: str
layer_idx: int
logical_block_idx: int
physical_block_id: int
is_null: bool
is_permanently_evicted: bool
was_compute_skipped: bool
score: float | None
ema_score: float | None
last_scored_step: int
last_selected_step: int
```

保留现在已有结构，删掉不精确或可推导字段，比如 `num_tokens`。当前 block token 数由 request length + block size 推导，不应在 block state 里长期维护一份近似值。

### `KVBlock` 是否维护稀疏状态

可以，但只能维护**物理块全局状态**，不能维护 request-specific 的可见性。

当前 `KVBlock` 是物理块对象，字段包括 `block_id/ref_cnt/block_hash/is_null`。同一个物理块可能被 prefix cache 命中后被多个 request 共享，所以这些状态适合放在 `KVBlock`：

```python
is_null
is_resident
summary_version
last_summary_step
```

这些状态不适合放在 `KVBlock`：

```python
request_id
layer_idx
logical_block_idx
was_compute_skipped
is_selected_for_next_step
```

原因：同一个 physical block 在不同 request 中可能对应不同 logical block index，甚至不同请求的 sparse plan 也不同。把 request/layer 可见性写进 `KVBlock` 会污染 prefix cache 共享语义。

推荐边界：

```text
KVBlock:
  physical block global state

BlockSparseState:
  request/layer/logical-block sparse state

KVBlockSummary:
  model runner side physical block summary
```

永久驱逐时，`req_to_blocks[logical_idx]` 替换为 `null_block`，被释放的原 physical block 可以继续保留 block_hash/prefix-cache entry，直到 BlockPool 重新分配该物理块时再按现有 `_maybe_evict_cached_block()` 清理。

### 2. `LayerSparsePlan`

位置：`kvcore/kv/sparse.py`

职责：下一次 forward 某个 request/layer 的 logical block 可见性计划。

字段建议：

```python
request_id: str
layer_idx: int
selected_block_indices: tuple[int, ...]
evicted_block_indices: tuple[int, ...] = ()
is_sparse: bool = False
```

不要同时保留 keep/skip/visible ids 三套表达。`skip` 可以由 full logical blocks - selected 推导；`visible block ids` 由 ModelRunner 物化。

`is_sparse=False` 表示该请求/层按 dense 方式读取 full logical block table。这样可以自然表达：

- batch 中只有部分请求稀疏；
- 某个请求完全不稀疏；
- prefill 默认不稀疏；
- decode 阶段根据计划稀疏。

### 3. `SparseKVPlan`

位置：`kvcore/kv/sparse.py`

职责：一整个 scheduler step 的 sparse plan。

字段建议：

```python
layer_plans: tuple[LayerSparsePlan, ...]
```

可以保留现在这个外壳。不要新增 `BlockVisibilityPlan`，避免多一层同义结构。

如果某个 request/layer 没有出现在 `SparseKVPlan` 中，默认 dense。这样无需为 dense 请求生成冗余 plan。

### 4. `KVBlockSummary`

位置：`kvcore/model/block_score.py` 或 `kvcore/model/block_summary.py`

职责：ModelRunner 侧的 GPU summary tensor，对 block score 计算服务。

字段建议：

```python
layer_idx: int
physical_block_id: int
mean_key: torch.Tensor
representative_keys: torch.Tensor
updated_step: int
```

是否包含 value summary 取决于当前 score 算法。当前你说通过 q 和 KV 块摘要计算，若只用 key 代表性即可，先不要加 value summary。

注意：summary 是 runner/model runtime 数据，不放进 `SchedulerOutput`，也不放进 `KVManager`。`KVManager` 只接收最终 score。

内存语义：`KVBlockSummary` 可以看作 KV tensor 的 mini 版，应覆盖所有未被永久驱逐、仍 resident 的 blocks。summary metadata 可以共享，但 summary tensor 的生命周期跟 physical block residency 绑定。

## Ownership Boundary

### Scheduler

职责：

- 调度请求和 token budget。
- 调用 `KVManager.allocate_slots()` 维护完整 logical block table。
- 在 `SchedulerOutput` 中携带 full block ids 和 `SparseKVPlan`。
- 在 `update_from_outputs()` 中接收 `BlockScoreUpdate`，交给 `KVManager` 更新 score，并生成下一轮 plan 所需状态。

不负责：

- 不生成 `visible_block_ids`。
- 不构造 attention metadata。
- 不理解 Triton block table 布局。

### KVManager

职责：

- 维护完整 logical block table。
- permanent eviction 时把对应 logical block 替换为 `null_block`，释放当前 request 对物理块的引用。
- 不删除 prefix cache 条目，不压缩 logical block index。
- 维护 `BlockSparseState` 和 EMA score。
- 根据 score + config 生成下一次 `SparseKVPlan`。

永久驱逐规则：

```text
logical block index stays
req_to_blocks[logical_idx] = null_block
physical block ref for this request is released
prefix cache mapping is not removed here
slot mapping for future new tokens still uses full logical positions
```

这就是借鉴 vLLM null block 思想的核心。

### SchedulerOutput

破坏式改成：

```python
@dataclass(frozen=True, slots=True)
class NewRequestData:
    req_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_params: SamplingParams
    block_ids: tuple[tuple[int, ...], ...]
    num_computed_tokens: int

@dataclass(frozen=True, slots=True)
class CachedRequestData:
    req_ids: tuple[str, ...]
    new_token_ids: tuple[tuple[int, ...], ...]
    block_ids: tuple[tuple[tuple[int, ...], ...], ...]
    num_computed_tokens: tuple[int, ...]
    num_output_tokens: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    scheduled_new_reqs: tuple[NewRequestData, ...]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    sparse_plan: SparseKVPlan
    finished_req_ids: frozenset[str] = frozenset()
    new_block_ids_to_zero: tuple[int, ...] = ()
    step_id: int = 0
```

删除：

```python
visible_block_ids
visible_block_indices
```

### ModelRunner / InputBatch

职责：

- 持久维护 request rows、token ids、full block ids。
- 根据 `SchedulerOutput.sparse_plan` 物化 sparse attention metadata。
- 使用 full block ids 计算 KV write slot mapping。
- 使用 selected block ids / selected logical indices 给 attention backend 做 sparse read。
- 维护 `KVBlockSummary`，计算 block score update。

`InputBatch` 不需要保存 scheduler 给的 visible table；它只保存 full request state。

`ModelRunner._build_attention_metadata()` 做四件事：

1. full block ids -> write slot mapping。
2. full block ids + sparse plan -> selected read block table。
3. dense 请求直接使用 full read table；sparse 请求使用 compact selected read table。
4. selected read block table + logical indices -> `PagedAttentionMetadata`。

### Attention Backend

允许改造后端，且建议直接改造 Triton backend。

runtime 只支持一个 backend：

```text
Triton paged attention with dense and sparse rows
```

不做：

- backend fallback
- capability routing
- torch runtime fallback

但测试中保留 `torch_paged` oracle。

后端必须支持同一个 batch 里 dense row 和 sparse row 混合。这个支持不是兼容包袱，而是功能需求。

## Sparse Attention Metadata

当前 `PagedAttentionMetadata` 可以破坏式改为包含 dense/sparse 混合所需字段：

```python
kv_cache_tensor: torch.Tensor
block_tables: MultiGroupBlockTable          # selected physical block ids
block_indices: dict[int, torch.Tensor]      # selected logical block indices per layer
row_is_sparse: torch.Tensor                 # [num_reqs], ModelRunner 预先构建
num_read_blocks: torch.Tensor               # [num_reqs, num_layers] 或 per-layer table
slot_mapping: dict[int, torch.Tensor]       # full write slot mapping
query_start_loc: torch.Tensor
seq_lens: torch.Tensor
context_lens: torch.Tensor
query_lens: torch.Tensor
flat_positions: torch.Tensor
token_request_indices: torch.Tensor
...
```

如果 `MultiGroupBlockTable` 已经内置 `block_indices`，就不要再加 `block_indices` 字段，避免重复。当前代码已经在 `BlockTable` 中加了 `block_indices` buffer，可以直接沿用。

原则：

- `block_tables` 表示 read table；dense row 是 full table，sparse row 是 compact selected table。
- `slot_mapping` 表示 full write mapping。
- `block_indices` 表示 read table 中每个 entry 对应的 original logical block index。
- `row_is_sparse` 和 `num_read_blocks` 由 ModelRunner 预先写好，kernel 不根据 mask 实时筛选。

## Triton Backend 改造

当前 dense kernel 逻辑大致是：

```text
for logical_block_idx in 0..query_position//block_size:
    physical_block = block_table[request_idx, logical_block_idx]
    read all offsets
```

改成 dense/sparse row 统一 kernel：

```text
num_blocks = num_read_blocks[request_idx]
for read_idx in 0..num_blocks:
    logical_block_idx = block_indices[request_idx, read_idx]
    physical_block = block_table[request_idx, read_idx]
    if logical block overlaps [0, query_position]:
        read valid token offsets
```

注意：

- 不读取 dense selected mask。
- sparse row 不遍历完整 logical block range。
- dense row 的 read table 由 ModelRunner 预先写成 full block table，kernel 仍走同一套 read loop。
- `query_position` 只用于判断 selected block 内哪些 token offset 有效。
- 最新写入 block 必须由 `SparseKVPlan` 保护，确保当前 token 的 K/V 已在 selected table 或 write slot mapping 不受影响。

这会增加一次 `block_indices` 读取，但省掉大量 skipped block 的 K/V 读取和 dot-product。对长上下文 decode，这是正确方向。

## Block Score / Summary Flow

### Summary 生命周期

`KVBlockSummary` 由 ModelRunner 维护，key 是：

```python
(layer_idx, physical_block_id)
```

更新时机：

- prefill 完成后，为 prefill 产生的 resident blocks 建 summary。
- 每步 decode 后，当前写入过的 block summary 可能需要刷新。
- 对已永久释放且不再 resident 的 block，summary 可以从 runner cache 删除。
- 对 dynamic skipped 但仍 resident 的 block，summary 保留。
- 当 physical block 被重新分配或清零时，summary 必须失效。

### Score 计算

`BlockScoreCollector` 改成：

1. attention layer 通过 `ForwardContext` 记录 post-RoPE query。
2. ModelRunner 根据 full block ids 找所有未永久驱逐、仍 resident 的候选 blocks。
3. 对候选 block 取 `KVBlockSummary`。summary 理论上覆盖所有未永久驱逐 blocks，而不只是本 step selected blocks。
4. 使用 q window 与 summary 计算 score。
5. 输出 `BlockScoreUpdate`。

`BlockScoreUpdate` 保持现在的形态即可：

```python
request_id
layer_idx
logical_block_indices
scores
score_kind
step_id
```

不要把 summary tensor 传给 scheduler/kvmanager。

## Execution Flow

### Step N schedule

```text
Scheduler.schedule()
  allocate full KV slots
  KVManager.plan_sparse_blocks(req, layer, context)
  return SchedulerOutput(full block_ids, sparse_plan)
```

### Step N model runner

```text
ModelRunner.execute_model(scheduler_output)
  update InputBatch full token/block state
  prepare input_ids/positions
  build full slot_mapping
  materialize dense/sparse read block table from sparse_plan
  run Triton paged attention with dense and sparse rows
  update KVBlockSummary
  compute BlockScoreUpdate
```

### Step N update

```text
ModelRunner.sample_tokens()
EngineCore.step()
Scheduler.update_from_outputs(sampled tokens, block_score_updates)
  KVManager.update_block_scores()
  KVManager may permanent-evict low-score blocks by replacing with null_block
  request state advances
```

### Step N+1

```text
KVManager uses updated scores/states to produce next SparseKVPlan
```

## Minimal Code Change Plan

### Phase 1: 删除重复 visible 字段

改动：

- `kvcore/sched/utils.py`
  - 删除 `visible_block_ids`
  - 删除 `visible_block_indices`
- `kvcore/sched/scheduler.py`
  - `build_sparse_plan()` 只返回 `SparseKVPlan`
  - `SchedulerOutput` 只带 full block ids + plan
- `kvcore/model/input_batch.py`
  - `CachedRequestState` 删除 visible 字段
  - `InputBatch.add_request/update_cached_request()` 只写 full block ids

验证：

- scheduler sparse plan test 改成只断言 `SparseKVPlan.selected_block_indices`。

### Phase 2: KVManager null block permanent eviction

改动：

- `SingleTypeKVManager.evict_blocks()`
  - 保留 logical index。
  - 替换 null block。
  - 释放当前 request block ref。
  - 不动 prefix cache 全局 mapping。
- `BlockSparseState`
  - 删除 `num_tokens`。
  - 增加 `is_null` 或直接由 `physical_block_id == NULL_BLOCK_ID` 推导。

验证：

- permanent eviction 后 `get_block_ids()` 长度不变。
- evicted logical index block id 为 0。
- 新 token slot mapping 不受旧 evicted block 影响。
- prefix cache 不因 eviction 被主动删除。

### Phase 3: ModelRunner 物化 selected read table

改动：

- `InputBatch` 只保留 full table。
- `ModelRunner._build_attention_metadata()` 每步根据 `SparseKVPlan` 写 dense/sparse mixed read `BlockTable`。
- `slot_mapping` 始终从 full block ids 构造。

验证：

- dense request 不出现在 plan 中时，read table 等于 full table。
- sparse plan `[1, 3]` 生成 selected physical ids 和 logical indices。
- 同 batch 内 dense row + sparse row 同时存在。
- write slot mapping 仍写到 full table 的当前 logical block。

### Phase 4: Triton dense/sparse paged attention

改动：

- 删除 `TritonPagedAttentionBackend._raise_if_sparse_read()`。
- 修改 read kernel：遍历 ModelRunner 预构建好的 read table。
- dense row 的 read table 是 full table；sparse row 的 read table 是 compact selected table。
- `torch_paged` oracle 保留同样语义。

验证：

- 单请求 decode sparse read 与 torch oracle 对齐。
- 多请求 decode sparse read 对齐。
- 同 batch 内部分请求 sparse、部分 dense 对齐。
- 同 batch 内全部请求 dense 对齐。
- GQA 对齐。
- dense selected full table 与旧 dense 输出对齐。

### Phase 5: Block summary + score

改动：

- 新增/收敛 `KVBlockSummary` 类。
- `BlockScoreCollector` 或 `BlockSummaryManager` 维护 summary cache。
- score update 只返回 logical block score。

验证：

- score update 覆盖所有 resident candidate blocks。
- dynamic skipped block 仍可保留 summary。
- permanent evicted block 不再参与后续 score。
- physical block 重分配后旧 summary 失效。

### Phase 6: Permanent eviction timing

改动：

- prefill 阶段完成后允许对 prefill 产生的 KV blocks 做一次 permanent eviction。
- decode 阶段主要做 compute sparsity。
- 只有生成长度很长、达到较大 sparse interval 时，再触发 permanent eviction。

建议默认：

```text
prefill_end_permanent_eviction = enabled
decode_compute_sparsity = enabled
decode_permanent_eviction_interval_tokens = large
```

验证：

- prefill 完成后低分 block 被替换为 null block。
- decode 前几步只做 compute sparsity，不释放物理块。
- 长 decode 到 interval 后才再次 permanent eviction。

## What To Delete

为保持项目精简，建议直接删除：

- `visible_block_ids`
- `visible_block_indices`
- runtime fallback 到 torch backend 的设想
- `SparseKVMode.HYBRID`
- 当前脚本里让用户误以为 hybrid/permanent 已完整实现的参数
- 任何只是为了兼容旧 `SchedulerOutput` 的 bridge code

`SparseKVMode` 可缩成：

```python
disabled
dynamic
permanent
```

其中：

- `dynamic`: compute sparse only，不释放物理块。
- `permanent`: compute sparse + 允许 KVManager 把低分 blocks 替换成 null block 并释放物理块。

永久驱逐时机不由 mode 名字隐含，单独由 config 控制：

```python
enable_prefill_eviction: bool
decode_eviction_interval_tokens: int
```

## Risks

### 1. Sparse Triton kernel 正确性风险

selected logical block indices 不连续后，kernel 必须正确处理：

- block 内 offset。
- query_position 截断。
- 当前写入 token。
- null block。

最小化方式：先只支持 decode sparse，prefill 保持 full。

### 2. Permanent eviction 与 prefix cache 的引用关系

如果 `BlockPool.free_blocks()` 会影响 prefix cache 中同一个 physical block 的可复用性，需要进一步检查 `BlockPool` 的 ref count/cache map 语义。目标不是删除 prefix cache entry，而是释放当前 request 对该 physical block 的引用。

### 3. Summary stale 问题

如果 physical block 被复用，旧 summary 必须按 `(layer_idx, physical_block_id, generation/version)` 或在 block zero/new allocation 时清除。当前最小实现可以在 `new_block_ids_to_zero` 时清除对应 summary。

### 4. Dense/sparse mixed batch 元数据一致性

一个 batch 中可能部分 request 稀疏、部分 dense。风险在于 row 的 `block_table/block_indices/num_read_blocks` 不一致。解决方式是所有 row 都使用同一套 read table 结构：

```text
dense row:
  read table = full logical blocks
  block_indices = 0..num_blocks-1

sparse row:
  read table = selected blocks
  block_indices = selected logical indices
```

kernel 只看 `num_read_blocks`，不做 mask 判断。

## Conclusion

新的破坏式方案是：

```text
Scheduler/KVManager:
  full token ids + full logical KV table + next SparseKVPlan
  permanent eviction = null block tombstone + physical ref release

ModelRunner:
  consume SparseKVPlan
  pre-materialize dense/sparse mixed read table
  compute full write slot mapping
  maintain KVBlockSummary for all resident non-evicted blocks and produce BlockScoreUpdate

Triton backend:
  natively iterate prebuilt read blocks
  no dense selected mask filtering
  no runtime fallback
```

这比上一版更贴近你的真实需求：KV 压缩是“不可见性”问题，permanent eviction 是“物理驻留”问题，二者都不应该破坏完整逻辑 KV 表。
