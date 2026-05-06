# vLLM Sliding Window Attention Notes for KV Sparse Refactor - 2026-05-06

## Problem Statement

当前 KVCore 的 sparse KV 原型已经能表达 `full block_ids + visible block_ids + sparse_plan`，但设计还显得不够自然。用户希望从 vLLM 对 sliding window attention 的实现里提炼更好的重构灵感，特别是让 attention kernel 的侵入尽可能小，同时让 scheduler/KV manager 维护完整 KV 信息。

本文基于 vLLM 当前文档/源码视图和 KVCore 当前未提交代码进行分析。

## vLLM Sliding Window 的核心实现

### 1. Sliding window 是 KV cache spec / manager 语义，不是普通 scheduler 字段

vLLM 把不同 attention 类型抽象成不同 KV cache spec/manager。`SlidingWindowManager` 继承自 `SingleTypeKVCacheManager`，持有 `sliding_window`，并按该窗口定义跳过 token 的规则。

关键点是 `get_num_skipped_tokens(num_computed_tokens)`：

```python
return max(0, num_computed_tokens - self.sliding_window + 1)
```

含义：对下一个 token 来说，窗口外的 prefix token 不再参与 attention 计算。来源：vLLM `single_type_kv_cache_manager.py` 文档中 `SlidingWindowManager.get_num_skipped_tokens()`。

这说明 vLLM 没有把 sliding window 当成一个临时的 `SchedulerOutput.visible_block_ids` 字段，而是把它放进 per-layer KV manager policy。

### 2. 窗口外 blocks 被替换为 null block，逻辑索引不压缩

`SingleTypeKVCacheManager.remove_skipped_blocks()` 的语义是：移除不再需要参与 attention 的 blocks，并用 `null_block` 替换原位置。vLLM 文档明确写到 removed blocks should be replaced by `null_block`，并且该函数依赖不同 attention type 自己实现的 `get_num_skipped_tokens()`。

这点非常关键：

- 物理块可以被释放。
- logical block index 不漂移。
- block table 仍保持原始逻辑位置。
- attention 侧看到的是带 null/tombstone 的逻辑表，而不是 scheduler 生成的 compact table。

这与我们当前 KV sparse 里“visible block ids + visible indices”很接近，但 vLLM 更偏向维护一张稳定逻辑表，而不是让 scheduler 产出压缩后的执行表。

### 3. Prefix cache 命中也按窗口语义定制

`SlidingWindowManager.find_longest_cache_hit()` 不是从左到右找最长 full prefix，而是从右往左找足够连续的命中块。原因是 sliding window 只需要最近窗口内的连续 KV。vLLM 的注释示例里，返回结果可以是类似 `[NULL, NULL, block8, block3]` 这样的形态：前面窗口外 blocks 是 null，后面窗口内 blocks 是真实 cache hit。

这说明：

- 对 local/window attention，prefix cache 不再是“从开头连续命中”的唯一形态。
- 但 vLLM 仍然用 null block 保护 logical position。
- cache manager 负责把“哪些 block 已经不需要真实物理存储”表达出来。

### 4. Hybrid KV manager 可禁用，但 compute 仍按 sliding window 做

vLLM 的 `FullAttentionSpec` 文档说明：当 hybrid allocator 被禁用，混合 full attention 和 sliding window attention 的模型会在 KV cache manager 中把 sliding window 当成 full attention，给所有 token 分配 blocks；但在 model runner 中仍然按 sliding window attention 计算。

`unify_hybrid_kv_cache_specs()` 也会把 `SlidingWindowSpec` 转成带 `sliding_window` 字段的 `FullAttentionSpec`，并警告这种模式不会做滑窗外 KV cache memory saving，但 compute saving 仍然存在。

这个设计把两种优化拆开：

- **Storage saving**：KV manager 是否释放窗口外物理块。
- **Compute saving**：model runner / attention backend 是否只计算窗口内 token。

这对 KV sparse 非常重要。我们也应该把 permanent eviction 和 dynamic compute sparsity 明确拆开，而不是把二者都揉进 `visible_block_ids`。

### 5. Attention layer 只保存 sliding_window 并传给 backend impl

vLLM `Attention` layer 初始化时接收 per-layer sliding window，并把 `sliding_window` 传给具体 backend implementation。backend impl 知道本层是否 local/window attention，但不负责选择哪些 block 被保留，也不负责维护 KV lifecycle。

换句话说，vLLM 的分层是：

```text
KV cache manager:
  own block lifecycle, skipped token policy, null block tombstones

Model runner / attention metadata:
  prepare runtime metadata and dispatch backend

Attention backend:
  consume window/local-attention parameters and block table
```

## 对 KVCore sparse KV 的启发

### 启发 1：不要把 sparse 先做成 compressed table，而要先做成 per-layer visibility policy

当前 KVCore 在 `Scheduler._try_schedule_request()` 中直接得到 `visible_block_ids` / `visible_block_indices`，再交给 `InputBatch`。这会让 scheduler 过早理解 runner-side metadata。

更像 vLLM 的做法应该是：

```text
KVManager:
  keeps full logical block table
  keeps BlockSparseState
  computes a SparseVisibilityPolicy/SparseKVPlan

SchedulerOutput:
  carries full block_ids + sparse_plan only

ModelRunner:
  materializes runtime table from full block_ids + sparse_plan
```

与上一版报告相比，我会进一步修正：`SparseKVPlan` 最好也不要设计成“压缩表生成指令”，而是设计成“每层逻辑 block 的可见性 mask/tombstone 状态”。

### 启发 2：KV sparse 应该分成三种语义，而不是一个 mode

vLLM sliding window 至少隐含三层：

1. **Logical visibility**：哪些 token/block 对 attention 可见。
2. **Physical residency**：哪些 block 仍持有真实物理 KV。
3. **Runtime table materialization**：backend 本 step 实际怎样读。

KVCore 当前 `dynamic/permanent/hybrid` 太粗。更清晰的命名应该是：

- `SparseVisibilityPolicy`: 生成可见 block mask，dynamic score / sink / recent 都属于这里。
- `SparseResidencyPolicy`: 决定是否把不可见或低分 block 永久释放。
- `SparseRuntimeLayout`: dense-with-null、compact-with-indices、backend-native sparse 等 runtime 形态。

这样不会出现 `hybrid` 看起来启用了 permanent eviction，但实现实际只做 dynamic compute 的问题。

### 启发 3：优先使用 tombstone/null-block 保持 logical block table 稳定

vLLM sliding window 的重要技巧是：窗口外 block 被释放后，原 logical block index 位置仍保留 null block。

对 KV sparse 来说，这比直接 compact 更优雅：

- Scheduler/KVManager 可以永远维护完整 logical block index。
- Prefix cache、debug、score state 不会因为压缩而发生索引漂移。
- Permanent eviction 可以自然表示为 tombstone。
- Attention metadata builder 可以选择 dense-with-null 或 compact runtime layout。

这也解释了为什么现在直接在 `SchedulerOutput` 塞 `visible_block_ids` 不够理想：它跳过了最核心的稳定逻辑表。

### 启发 4：compute sparse 和 memory sparse 可以独立退化

vLLM 在禁用 hybrid KV manager 时仍保留 sliding-window compute，这给 KVCore 一个很好的退化策略：

- 如果 backend 不支持 compact sparse read，可以先用 dense table + mask/metadata 做 correctness，或者直接禁用 compute sparse。
- 如果 memory pressure 不要求永久释放，可以只做 dynamic compute sparse，不改物理 residency。
- 如果要节省 KV memory，再启用 residency policy，把不可见低分 block 替换成 null block 并释放物理块。

这比当前 `mode=dynamic/permanent/hybrid` 更可控。

## 更推荐的 KVCore 重构方向

### 1. 把 sparse 状态并入 KV manager 的 block table 语义

当前 `SingleTypeKVManager` 已经有：

- `req_to_blocks`
- `permanently_evicted_blocks`
- `block_sparse_states`
- `SlidingWindowKVManager.get_num_skipped_tokens()`

这非常适合扩展成 vLLM 风格：

```python
class SingleTypeKVManager:
    def get_num_skipped_tokens(...)
    def remove_skipped_blocks(...)
    def get_block_visibility(...)
    def get_runtime_blocks(...)
```

其中：

- full attention: skipped tokens = 0。
- sliding window: skipped tokens = `max(0, computed - window + 1)`。
- sparse dynamic: skipped blocks 来自 score policy，但默认不释放。
- sparse permanent: skipped blocks 可替换成 null block 并释放。

### 2. 用 `BlockVisibilityPlan` 替代 `visible_block_ids`

推荐新数据结构：

```python
@dataclass(frozen=True, slots=True)
class LayerBlockVisibility:
    request_id: str
    layer_idx: int
    visible_logical_indices: tuple[int, ...]
    resident_logical_indices: tuple[int, ...]
    skipped_logical_indices: tuple[int, ...]
    evicted_logical_indices: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class BlockVisibilityPlan:
    layers: tuple[LayerBlockVisibility, ...]
```

`SchedulerOutput` 只传：

```python
block_ids              # full logical table, null block included
block_visibility_plan  # logical visibility, not physical compact table
```

不传：

```python
visible_block_ids
visible_block_indices
```

### 3. Runtime layout 由 ModelRunner 选择

`ModelRunner` 可以根据 backend capability 选择：

```text
dense_with_null:
  full logical table, skipped positions point to null / masked out

compact_with_indices:
  compact physical block ids + logical_block_indices side table

backend_native:
  future Triton/FlashInfer custom layout
```

这比 scheduler 直接生成 visible table 更优雅，因为同一个 visibility plan 可以被不同 backend materialize 成不同 runtime layout。

### 4. Attention backend 不接触 sparse policy

backend 应该只声明 capability：

```python
supports_dense_null_blocks = True
supports_compact_block_indices = False
supports_block_visibility_mask = False
```

进入 backend 前由 `Attention` wrapper 或 metadata builder 做检查。这样 Triton 不需要实现 `_raise_if_sparse_read(block_indices)` 这种 policy 判断；它只说自己支持哪种 layout。

### 5. Score collector 不应该直接绑定最终 read layout

`BlockScoreCollector` 继续通过 `ForwardContext` 捕获 post-RoPE query，计算 block score，传回 `KVManager`。但 score update 只更新 `BlockSparseState`，不直接产出 visible table。

这与 vLLM 的思路一致：attention layer 可以暴露必要 runtime 信息，但 KV lifecycle/policy 仍回到 manager 层。

## 对当前方案的具体修正

### 应该保留

- `BlockSparseState`：方向正确。
- `full write slot mapping` 和 `read table` 分离：方向正确。
- `torch_paged` 做 correctness oracle：方向正确。
- `triton_paged` 对 unsupported sparse layout 不静默执行：方向正确。

### 应该调整

1. `KVManager.build_sparse_plan()` 不要返回 `visible_block_ids`，只返回 logical visibility。
2. `SchedulerOutput` 删除 `visible_block_ids` / `visible_block_indices`。
3. `InputBatch` 不应该存 scheduler 给的 visible table，而应存 full logical table + latest visibility plan。
4. `BlockTable` 改成 runtime materialization target，不是 sparse policy container。
5. `SparseKVMode` 拆成 visibility/residency/runtime 三个维度。
6. `SlidingWindowKVManager` 当前已有 `get_num_skipped_tokens()`，应补 `remove_skipped_blocks()`，并作为 sparse permanent/tombstone 的参考实现。

## Recommended Next Design

更好的目标架构可以写成：

```text
Scheduler
  schedule tokens
  ask KVManager for full block ids and BlockVisibilityPlan

KVManager / SingleTypeKVManager
  own full logical block table
  own block score states
  own tombstone/null block replacement
  implement visibility/residency policy

ModelRunner / InputBatch
  own persistent request rows
  keep full logical table snapshot
  materialize AttentionRuntimeLayout per backend

Attention metadata
  kv_cache_tensor
  slot_mapping from full logical table
  read_layout = dense_with_null | compact_with_indices
  optional logical_block_indices side table

Attention backend
  only consume supported read_layout
```

## Minimal Validation Plan

1. **Sliding-window parity test**
   - 构造 block size 2、window 4、computed 7。
   - 验证 `get_num_skipped_tokens(7) == 4`。
   - 验证 skipped full blocks 被置 null，logical indices 不移动。

2. **Sparse visibility test**
   - 给定 full block ids `[1, 2, 3, 4]` 和 keep `{1, 3}`。
   - `BlockVisibilityPlan` 保持 logical indices `{1, 3}`。
   - dense-with-null layout 与 compact-with-indices layout 读到同一组 tokens。

3. **Residency separation test**
   - dynamic compute sparse 不释放物理块。
   - permanent sparse 把对应 logical index 替换成 null block 并 free block。

4. **Backend capability test**
   - torch backend 支持 compact oracle。
   - triton backend 不支持 compact 时在 wrapper 层报错。
   - dense full attention 不受影响。

## Conclusion

vLLM sliding window 最值得借鉴的不是具体公式，而是分层方式：

- 用 attention/KV spec 表达每层策略。
- 用 KV manager 维护完整逻辑表和 null block tombstone。
- 把 storage saving 和 compute saving 拆开。
- 让 model runner 决定 runtime metadata layout。
- backend 只消费已声明支持的 layout。

因此，KVCore 的 sparse KV 重构不应该继续围绕 `visible_block_ids` 做增量修补，而应转向 `full logical table + BlockVisibilityPlan + runtime layout materializer`。这会比当前方案更接近 vLLM 的抽象，也更适合后续接 Triton/FlashAttention。

## Sources

- vLLM `SingleTypeKVCacheManager.remove_skipped_blocks()` and `SlidingWindowManager`: https://docs.vllm.ai/en/stable/api/vllm/v1/core/single_type_kv_cache_manager/
- vLLM `FullAttentionSpec` sliding-window fallback behavior: https://vllm.website.cncfstack.com/api/vllm/v1/kv_cache_interface.html
- vLLM `unify_hybrid_kv_cache_specs()` hybrid manager fallback: https://docs.vllm.ai/en/stable/api/vllm/v1/core/kv_cache_utils/
- vLLM `Attention` layer passes sliding window into backend implementation: https://vllm.website.cncfstack.com/api/vllm/model_executor/layers/attention/attention/
