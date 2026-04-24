2026-04-23 22:53:56 CST

# KVManager / SingleTypeKVManager 端到端可运行性检查

## 检查目标

这里的“最小端到端测试”我按下面这个标准来判断：

1. 单进程、decoder-only、full attention。
2. `Scheduler` 持有 `KVManager`，`ModelRunner` 持有共享物理 KV tensor。
3. 至少能跑通：
   `Request -> KVManager.allocate_slots -> ModelRunner.build_attention_metadata -> model forward -> logits -> sample -> 下一步 decode 复用 KV`
4. 不要求：
   prefix cache 命中优化、投机解码、PP/TP、滑窗 attention、多模态。

如果只要求“模型能前向 + KVManager 能分块”，那现在基本够。
如果要求“真的带 KV cache 读写和下一步 decode 复用”，当前还**不够**。

## vLLM 参考接口

参考上游最新主线：

- `vllm/v1/core/kv_cache_manager.py`
  https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/kv_cache_manager.py
- `vllm/v1/core/single_type_kv_cache_manager.py`
  https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/single_type_kv_cache_manager.py

从职责上看，vLLM 的分层是：

- `KVCacheManager` 负责跨 layer/type 的统一入口。
- `SingleTypeKVCacheManager` 负责某一类 KV cache 的 block 生命周期。
- worker/model runner 负责把 block table / slot mapping / kv tensor 真正喂给 attention backend。

你现在把 coordinator 去掉、让每层单独逻辑管理，同时所有层共享一个大的物理 tensor，这个方向本身没有问题；关键在于 **逻辑 block 生命周期** 和 **运行时 attention 消费这些信息** 两边都要闭环。

## 目前已经具备的最小能力

### 1. `KVManager` 主入口接口基本齐了

本地 `kvcore/kv/kv_manager.py` 已具备这些接口：

- `get_computed_blocks`
- `can_fit`
- `allocate_slots`
- `cache_blocks`
- `free`
- `get_blocks`
- `get_block_ids`
- `take_new_block_ids`
- `evict_request_blocks`

这已经覆盖了最核心的 block 生命周期入口，和 vLLM 的主干接口是同一方向的。

### 2. `SingleTypeKVManager` 已经能支撑 full attention 的块管理

本地 `kvcore/kv/single_type_kv_manager.py` 已具备：

- `get_num_blocks_to_allocate`
- `allocate_new_computed_blocks`
- `allocate_new_blocks`
- `cache_blocks`
- `free`
- `evict_blocks`
- `take_new_block_ids`
- `find_longest_cache_hit`（full attention 已实现）

对 full attention 来说，这已经足够做：

- 命中 prefix cache 后挂接已有 block
- 分配新 block
- 释放 block
- 永久驱逐部分逻辑块并替换成 `null_block`

### 3. ModelRunner 已经能构建运行时 metadata

本地 `kvcore/model/model_runner.py` 已经能：

- 初始化共享物理 KV tensor
- 从 `KVManager` 读取 block ids
- 构建 `MultiGroupBlockTable`
- 构建 `slot_mapping`
- 生成 `KVForwardMetadata`

这意味着“调度侧逻辑块表 -> 运行时 metadata”这一步已经有骨架了。

## 和 vLLM 相比，当前还缺但暂时不阻塞最小 full-attention E2E 的点

这些差异存在，但如果我们只追求第一版 full-attention、单请求生成闭环，它们不是第一优先级阻塞项。

### 1. `can_fit` 语义比 vLLM 更弱

vLLM 里更接近 `can_fit_full_sequence` 的语义，会更强调 admission control 的“整条请求能否放下”。

你这里的：

- `can_fit(request, num_new_tokens, new_computed_blocks=None)`

更像“当前这一步还能不能再分一些块”。

对于单请求 smoke test 问题不大，但对真实 scheduler 来说，后面需要补强。

### 2. prefix cache 的默认接线还不完整

`Request` 已经支持 `block_hasher`，`kvcore/kv/kv_utils.py` 也提供了 `get_request_block_hasher`，但现在 scheduler 并没有统一替 request 自动注入 block hasher。

这意味着：

- prefix cache 机制本身不是没有；
- 但默认运行路径还没有把它稳定接上。

这不影响“无 prefix cache 的最小 E2E”，但影响“带 cache hit 的 E2E”。

### 3. sliding window 相关接口只保留了扩展位

`SlidingWindowKVManager.find_longest_cache_hit()` 还没做，`get_num_skipped_tokens()` 也还只是简单占位。

这和你当前阶段目标一致，不影响 full attention 最小链路。

## 真正阻塞“带 KV 复用的端到端测试”的点

下面这些是我认为**当前最关键的阻塞项**。

### 1. attention backend 还没有真正消费 `kv_cache_tensor` / `block_tables` / `slot_mapping`

这是当前最大的断点。

`kvcore/model/model_runner.py` 会生成：

- `KVForwardMetadata.block_tables`
- `KVForwardMetadata.slot_mapping`
- `KVForwardMetadata.kv_cache_tensor`

但是 `kvcore/model/attn_backend/torch_sdpa.py` 里当前实际做的是：

- 如果 `kv_cache` 对象有 `update()`，就调用它；
- 否则直接对本步的 `query/key/value` 做普通 `scaled_dot_product_attention`。

也就是说当前 backend：

- 不会把 `key/value` 写入共享物理 KV tensor；
- 不会根据 block table 从共享 KV tensor 中 gather 旧 K/V；
- 不会利用 slot mapping 做 paged write；
- decode 时也不会读取历史 cache。

结论：  
现在的 metadata 虽然造出来了，但 attention 计算并**没有真正用它**。

### 2. 模型 forward 仍然保留了旧的 `kv_caches: list[...]` 接口路径

例如 `kvcore/model/models/llama3.py` 仍然是：

- `forward(..., attn_metadata=None, kv_caches=None)`

layer 内部也还是把每层 `kv_cache` 显式传到 attention wrapper。

而你现在的架构已经改成：

- 单一共享物理 KV tensor
- 不再给每层绑定单独 cache 对象

这两套模型接口目前是并存但没有真正统一的。  
结果就是：

- 旧测试还能靠 `RecordingKVCache.update()` 跑通；
- 新的共享 tensor 路径却还没有真正接到 attention backend。

### 3. ModelRunner 还没有形成真正的“执行一步推理”的闭环 API

当前 `ModelRunner` 主要提供的是：

- `create_model`
- `load_model`
- `initialize_kv_cache`
- `build_attention_metadata`

但还缺一个明确的执行闭环，例如：

- `execute_prefill(...)`
- `execute_decode_step(...)`
- 或者统一的 `execute_model(...)`

这个 API 至少需要串起：

1. 从 request batch 组织 `input_ids` / `positions`
2. 调 `kv_manager.get_computed_blocks` / `allocate_slots`
3. 构建 `attn_metadata`
4. 调模型 forward
5. `compute_logits`
6. sampler 采样
7. 回写 `request.append_output_token_ids(...)`
8. 更新 `request.num_computed_tokens`

现在这些零件分散存在，但没有真正连成一条运行路径。

### 4. `request.num_computed_tokens` 的推进还没有被统一接管

`KVManager.allocate_slots()` 的分配逻辑依赖：

- `request.num_computed_tokens`

但当前没有看到一个统一的 scheduler/model runner 路径，在每步 forward 完成后稳定地更新这个值。

这会直接影响：

- 下一步需要分配多少 block
- prefix/cached block 对齐是否正确
- decode 阶段逻辑块表是否增长正确

如果没有这一步，KVManager 的接口即使齐了，也只是“静态可调用”，还不是“动态可推进”。

## 结论

### 结论一：如果目标只是静态 smoke test

也就是：

- 单轮 prefill
- 不依赖历史 KV
- 不验证 decode 复用
- 只验证 `KVManager` 分块、`ModelRunner` 产 metadata、模型能 forward

那么当前实现**基本已经够了**。

### 结论二：如果目标是真正的生成式端到端测试

也就是：

- prefill 后继续 decode
- 下一步能复用先前写入的 KV
- attention backend 真的从共享大 tensor 读写 cache

那么当前实现**还不具备最小要求**。

核心原因不是 `KVManager` / `SingleTypeKVManager` 的接口不够，而是：

1. runtime metadata 还没有被 attention backend 真正消费；
2. 共享物理 KV tensor 路径还没替代旧的 per-layer `kv_caches.update()` 路径；
3. 缺少一个统一的 step-level 执行 API 去推进 request / kv / sampling 状态。

## 我建议的最小补全顺序

按最短闭环来做，我建议顺序是：

1. 在 `ModelRunner` 增加一个最小 `execute_model_step(...)` 或 `execute_decode_step(...)`
   - 先只支持单 batch、decoder-only、full attention。

2. 统一模型/attention 接口
   - 不再依赖 `kv_caches: list[object]`
   - attention backend 只从 `attn_metadata` 读取 `kv_cache_tensor + block_tables + slot_mapping + layer_idx`

3. 在 eager backend 先做一个最小版“paged KV 读写”
   - 本步 `key/value` 按 `slot_mapping` 写入共享 tensor
   - decode 时按 block ids 从共享 tensor gather 历史 K/V
   - 即使先不用高性能 kernel，也要先把语义跑通

4. 把 `request.num_computed_tokens` 的推进放到 scheduler/model runner 主路径中
   - prefill 完成后更新
   - 每次 decode 产出 token 后继续更新

5. 最后再补 prefix cache 自动接线
   - scheduler 创建 request 时自动注入 `get_request_block_hasher(block_size)`

## 最小判断表

| 项目 | 当前状态 | 是否满足真实 KV E2E |
| --- | --- | --- |
| `KVManager` 主入口接口 | 已具备 | 是 |
| `SingleTypeKVManager` full attention 生命周期 | 已具备 | 是 |
| 共享物理 KV tensor 初始化 | 已具备 | 是 |
| block table / slot mapping / metadata 构建 | 已具备 | 是 |
| attention backend 写入共享 KV tensor | 未具备 | 否 |
| attention backend 从共享 KV tensor 读取历史 KV | 未具备 | 否 |
| 模型接口完全切换到 metadata 路径 | 未具备 | 否 |
| step-level 推理执行闭环 | 未具备 | 否 |
| `num_computed_tokens` 自动推进 | 未具备 | 否 |

## 最终判断

**当前 `KVManager` / `SingleTypeKVManager` 的接口，已经基本满足“最小逻辑块生命周期管理”的要求；但还没有满足“真实带 KV cache 复用的端到端测试”的最小要求。**

如果你愿意，下一步我建议直接补第一个真正的闭环：

- 先实现 `ModelRunner.execute_model_step(...)`
- 再把 `TorchSDPAAttentionBackend` 改成最小可用的共享 KV tensor 读写版本

这样一做，后面很多讨论就会从“接口是否足够”变成“具体语义是否正确”，推进会快很多。
