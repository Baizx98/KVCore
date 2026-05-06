# KV Sparse Architecture Review - 2026-05-06

## Problem Statement

当前未提交代码已经实现一条 dynamic sparse KV 原型链路：`ModelRunner` 收集 query-aware block score，`EngineCore` 把 score 反馈给 `Scheduler`，`KVManager` 维护 per-request/per-layer block sparse state，下一步 `schedule()` 生成 visible block table 给模型执行。

本次审核关注三个目标：

1. `Scheduler/KVManager` 维护完整 KV 信息和稀疏策略状态。
2. `ModelRunner` 构建 attention metadata 时得到压缩后的 read block table。
3. 对 attention backend / kernel 的侵入尽可能小。

## Confirmed Current Shape

### 已经做对的方向

- 配置入口集中在 `SparseKVConfig`，默认关闭，decode sparsity 默认打开、prefill 默认关闭，符合先做 decode-only correctness 的策略。证据：`kvcore/config.py:96-157`。
- `KVManager` 已开始维护 `BlockSparseState`，包括 score、EMA、永久驱逐、dynamic skip 和 step 记录。证据：`kvcore/kv/sparse.py:20-32`、`kvcore/kv/single_type_kv_manager.py:96-103`。
- `SchedulerOutput` 同时保留 full `block_ids` 和 visible block 信息。证据：`kvcore/sched/utils.py:10-18`、`kvcore/sched/utils.py:21-29`。
- `InputBatch` 已拆出 `full_block_table` 和 `block_table`：前者用于 full slot mapping 写 KV，后者用于 sparse read。证据：`kvcore/model/input_batch.py:100-117`、`kvcore/model/model_runner.py:559-574`。
- `torch_paged` 已作为 correctness oracle 支持 non-contiguous sparse read block table。证据：`kvcore/model/attn_backend/torch_paged.py:183-200`。
- `triton_paged` 目前显式拒绝 sparse read，避免静默错误。证据：`kvcore/model/attn_backend/triton_paged.py:241-245`、`kvcore/model/attn_backend/triton_paged.py:317-325`。

### 当前主要问题

#### 1. SchedulerOutput 携带了过多 runner-side table 细节

当前 `Scheduler` 在 `_try_schedule_request()` 里不仅生成 full `block_ids`，还直接生成 `visible_block_ids` 和 `visible_block_indices`，并塞入 `NewRequestData` / `CachedRequestData`。证据：`kvcore/sched/scheduler.py:477-510`。

这能跑通原型，但边界不够优雅：

- `visible_block_ids` 是 attention metadata 的物化形态，不是 scheduler 的核心调度语义。
- `SchedulerOutput` 里同时放 full IDs、visible IDs、visible indices、`sparse_plan`，同一事实被重复表达，未来容易不一致。
- 如果后续支持不同 backend 的 metadata 形态，scheduler 会被迫理解更多 runner/backend 细节。

更推荐：`SchedulerOutput` 只保留 full `block_ids` + `sparse_plan`。`sparse_plan` 描述 keep/skip logical block indices，`ModelRunner/InputBatch` 根据 full IDs 和 plan 物化 compressed read block table。

#### 2. KVManager 同时承担 state store、policy planner、table materializer

`KVManager.build_sparse_plan()` 当前做了三件事：读取 full block IDs、按 score 选择 visible indices、直接生成 visible block IDs。证据：`kvcore/kv/kv_manager.py:282-412`。

这会让 `KVManager` 太重：

- KV lifecycle owner 的职责是维护完整逻辑块、物理块、prefix cache、永久驱逐和 sparse state。
- 稀疏选择策略更像 `SparseKVPlanner`：输入 block states + config + request context，输出 `SparseKVPlan`。
- visible block table 是 model runtime metadata，应该在 `ModelRunner` builder 中物化。

更推荐拆成：

- `SparseKVStateStore`：可以先继续放在 `SingleTypeKVManager` 内，维护 `BlockSparseState`。
- `SparseKVPlanner`：纯策略模块，输出 `SparseKVPlan`，不返回 block IDs。
- `SparseBlockTableBuilder`：runner 侧从 full block IDs + plan 生成 read block table。

这样 scheduler 仍然能维护完整 KV 信息，但不会变成 backend metadata builder。

#### 3. `BlockTable` 同时表示 full write table 和 sparse read table，语义过载

当前 `InputBatch` 持有两个 `MultiGroupBlockTable`：`full_block_table` 和 `block_table`。证据：`kvcore/model/input_batch.py:100-117`。

这个拆分方向正确，但命名和抽象还不够清楚：

- `block_table` 实际是 `read_block_table`。
- `full_block_table` 主要用于 `compute_slot_mapping()`，并不一定需要完整 GPU staged table。
- 两份完整 `MultiGroupBlockTable` 会把 row buffer / GPU buffer / slot buffer 都复制一遍，KV block 数很大时会明显增加 metadata 显存和 CPU pinned memory。

更推荐：

- `InputBatch.full_block_ids`: CPU-side full logical table state，作为 runner 持久状态。
- `InputBatch.read_block_table`: 本 step compressed read table，commit 到 GPU。
- `SlotMappingBuilder`: 用 full block IDs 计算 write slot mapping，不需要复用一整份 `MultiGroupBlockTable`。

如果为了短期少改，可以先把字段改名为 `read_block_table` / `write_slot_table`，并在注释里固定语义。

#### 4. Sparse 语义已经泄漏进 Triton backend

`TritonPagedAttentionBackend.forward()` 读取 `block_indices` 并调用 `_raise_if_sparse_read()`。证据：`kvcore/model/attn_backend/triton_paged.py:241-245`。

这比静默错算好，但仍然不是最小侵入：

- backend 现在必须知道 sparse read 的判定方式。
- 后续如果增加 FlashAttention 或另一个 kernel，每个 backend 都要复制类似检查。

更推荐在 `Attention` wrapper 或 `ModelRunner` metadata builder 层做 capability routing：

- `PagedAttentionMetadata` 显式标记 `read_table_kind = dense | sparse_compact`。
- backend 暴露 `supports_sparse_read`。
- 如果 backend 不支持 sparse read，在进入 backend 前报错或 fallback 到 `torch_paged`。

这样 kernel 只消费它支持的 metadata，不负责解释 sparse policy。

#### 5. 当前 config 暴露了 `permanent/hybrid`，但实现主要是 dynamic compute sparsity

`SparseKVMode` 支持 `PERMANENT`、`DYNAMIC`、`HYBRID`。证据：`kvcore/kv/sparse.py:7-11`。但 `KVManager.build_sparse_plan()` 对所有非 disabled 模式都走 dynamic visible-block selection；没有看到 mode-specific permanent eviction trigger。证据：`kvcore/kv/kv_manager.py:282-412`。

短期建议：

- 如果这轮只完成 dynamic compute sparsity，把 `permanent/hybrid` 标成 planned 或先不在 CLI 暴露。
- 如果保留 config，`build_sparse_plan()` 至少应按 mode 分支，避免用户以为 hybrid 已经生效。

#### 6. Step id 有两个来源，长期可能漂移

`KVManager.step_id` 在 `Scheduler.update_from_outputs()` 末尾推进。证据：`kvcore/sched/scheduler.py:403`。`BlockScoreCollector.step_id` 则在 `collect()` 内部自增。证据：`kvcore/model/block_score.py:17-19`、`kvcore/model/block_score.py:77`。

当前单 engine 同步路径下大概率一致，但这不是强约束。更稳的做法是：

- `Scheduler` 在 `SchedulerOutput` 中带 `step_id`。
- `ModelRunner` 生成的 `BlockScoreUpdate.step_id` 使用 `scheduler_output.step_id`。
- `KVManager.advance_step()` 只由 scheduler 统一推进。

## Recommended Architecture

### Contract

建议把主链路收敛成：

```text
Scheduler/KVManager
  owns full logical KV blocks + sparse states
  emits full block_ids + SparseKVPlan

ModelRunner/InputBatch
  owns runtime batch state
  materializes read_block_table from full block_ids + SparseKVPlan
  computes write slot_mapping from full block_ids

Attention backend
  consumes PagedAttentionMetadata
  does not own sparse policy
```

### Data Shape

推荐保留三个概念，避免混用：

- `full_block_ids`: 完整 logical block index -> physical block id，用于 KV 生命周期和 write slot mapping。
- `SparseKVPlan`: 每个 request/layer 的 keep/skip logical block indices。
- `read_block_table`: compact physical block IDs + logical block indices，用于 sparse attention read。

`SchedulerOutput` 建议只保留：

```python
NewRequestData.block_ids
CachedRequestData.block_ids
SchedulerOutput.sparse_plan
SchedulerOutput.step_id
```

删除或下沉：

```python
NewRequestData.visible_block_ids
NewRequestData.visible_block_indices
CachedRequestData.visible_block_ids
CachedRequestData.visible_block_indices
```

这些字段由 `ModelRunner` 从 `block_ids + sparse_plan` 生成。

### Minimal Kernel Intrusion

稀疏 read 对 kernel 不可能完全零侵入，因为 dense paged attention 默认按 `position // block_size` 找块；compact sparse read 需要知道 logical block index 到 compact table row 的映射。

但可以把侵入限制在一个统一 metadata contract：

- backend 不接触 `SparseKVPlan`。
- backend 只看到 `read_block_table` 和 `read_block_indices`。
- backend capability 由 wrapper 层判断。
- scoring 不从 attention weights 返回，继续独立用 summary score 计算，满足后续 FlashAttention 路径。

对 `torch_paged`：继续保留当前 oracle 逻辑。

对 `triton_paged`：短期在 wrapper 层拒绝 sparse metadata；长期只改 read kernel 的 block iteration，让它按 `read_block_indices` 跳过不可见 logical blocks。

## Minimal Refactor Plan

### Step 1: 收窄 SchedulerOutput

问题：scheduler 输出 runner-ready visible table。

方法：

- 保留 `SparseKVPlan`。
- 删除 `visible_block_ids` / `visible_block_indices` 字段。
- `KVManager.build_sparse_plan()` 改名为 `plan_sparse_blocks()`，只返回 `SparseKVPlan`。

预期收益：scheduler 不再绑定 attention metadata 形态。

风险：需要同步改 `InputBatch.update_cached_request()` 和相关测试。

最小验证：

- `tests/test_scheduler.py::test_scheduler_dynamic_sparse_plan_applies_to_decode_only`
- 新增断言：scheduler output 只有 full block IDs，keep/skip 在 `sparse_plan`。

### Step 2: 引入 runner-side SparseBlockTableBuilder

问题：InputBatch 现在接收 visible IDs。

方法：

- 新增 `kvcore/model/attention_metadata_builder.py` 或放在 `model_runner.py` 私有 helper。
- 输入：`InputBatch` full block IDs + `SparseKVPlan` + active req ids。
- 输出：`read_block_table` 和 full slot mapping。

预期收益：ModelRunner 才是 compressed block table 的唯一物化点。

风险：需要处理 batch row reorder 和多层 sparse plan 查找。

最小验证：

- dense plan 等价当前 full block table。
- sparse plan 对 non-contiguous logical indices 生成正确 `read_block_indices`。
- `torch_paged` sparse oracle 仍过。

### Step 3: 改名并瘦身 BlockTable

问题：两份 `MultiGroupBlockTable` 内存开销和语义不清。

方法：

- `input_batch.block_table` 改名 `read_block_table`。
- `input_batch.full_block_table` 短期改名 `write_block_table`；中期替换成 CPU full row + slot mapping builder。

预期收益：读写语义清楚，后续减少 metadata 内存。

风险：机械改名影响测试较多，但行为简单。

最小验证：

- `tests/test_block_table.py`
- `tests/test_torch_paged_attention.py`

### Step 4: Backend capability routing

问题：Triton backend 内部判断 sparse read。

方法：

- 在 backend 类上加 `supports_sparse_read: bool`。
- `Attention.forward()` 或 `ModelRunner._build_attention_metadata()` 根据 `metadata.is_sparse_read` 和 backend capability 报错或 fallback。
- 删除 `TritonPagedAttentionBackend._raise_if_sparse_read()`。

预期收益：backend/kernel 不需要理解 sparse policy，只声明能力。

风险：如果 fallback 到 torch，需要明确性能和 dtype 限制；正式实验时建议直接报错。

最小验证：

- dense triton path 不受影响。
- sparse + triton 在 wrapper 层给清晰错误。

### Step 5: 明确 mode 语义

问题：`permanent/hybrid` 暴露但未完整实现。

方法：

- dynamic-only 阶段：CLI 和文档只推荐 `mode=dynamic`。
- `permanent/hybrid` 分支先 raise `NotImplementedError`，或在文档中标注 planned。
- 后续 permanent eviction 由 `Scheduler.update_from_outputs()` 或专门 compression policy 在完整 block 边界触发。

预期收益：实验语义不混淆。

风险：如果已有脚本使用 `hybrid`，需要改默认参数。

最小验证：

- config validation test。
- CLI smoke test。

## Suggested Target File Ownership

- `kvcore/kv/sparse.py`: sparse dataclasses and pure planner types。
- `kvcore/kv/kv_manager.py`: full KV state + score update storage，不物化 visible block IDs。
- `kvcore/sched/scheduler.py`: schedule full blocks + attach `SparseKVPlan`。
- `kvcore/model/input_batch.py`: full request state + read block table state。
- `kvcore/model/model_runner.py`: attention metadata builder entrypoint。
- `kvcore/model/attn_backend/*`: only consume supported metadata。

## Validation Limits

本次尝试运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_block_table.py tests/test_scheduler.py tests/test_torch_paged_attention.py tests/test_model_runner_kv_boundary.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_block_table.py tests/test_scheduler.py tests/test_torch_paged_attention.py -q
git diff --check
```

结果：

- `git diff --check` 通过。
- 两次 pytest 都超过 30 秒无输出，子进程进入 `D` 状态后手动清理。这里不能作为功能失败证据，只能说明当前环境下这组测试无法给出完整验证结果。

## Conclusion

当前实现方向是可用的原型，但还没有达到最优雅边界。最关键的调整是：让 `Scheduler/KVManager` 只输出 full KV + `SparseKVPlan`，让 `ModelRunner` 独立物化 compressed read block table，并通过 backend capability 控制 sparse metadata 是否允许进入具体 kernel。

这样可以同时满足两个目标：

- scheduler 维护完整 KV 信息和稀疏状态；
- attention kernel 只面对统一 metadata，不知道 scheduler policy，也不直接处理 score/selection 逻辑。
