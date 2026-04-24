# Current Architecture And Flow

Date: 2026-04-23 17:43:52 CST

## Overview

KVCore 当前已经形成了一个 vLLM 风格但更精简的推理骨架：

- `utils` 定义请求和采样参数。
- `model_loader` 负责 HF 配置、模型骨架创建和权重加载。
- `models` 定义 Llama3/Qwen3/Mistral 的 decoder-only 模型结构。
- `layer.attention` 是模型层到 attention backend 的统一 PyTorch wrapper。
- `kv` 负责 KV block 生命周期、prefix cache、永久驱逐和 block table。
- `model_runner` 是运行时边界，负责 KV tensor 初始化、绑定、block table/slot mapping/attention metadata 构造。

当前已经有最小 scheduler/engine 执行循环：

- `EngineCore.step()` 串起调度、模型执行、采样和请求状态更新。
- `Scheduler` 负责 waiting/running 队列、prefill continuation、decode priority 和逻辑 KV block 分配。
- `ModelRunner.execute_model()` 负责构造 flattened runtime input、paged attention metadata、模型 forward、logits 和 sampler。

仍未完成的是高覆盖真实模型验证、性能 profiling、复杂 preemption/offload，以及与 vLLM 生产路径同等级的 kernel 优化。

## Request And Sampling

`kvcore/utils/request.py` 提供最小 `Request`：

- 保存 `request_id`、`prompt_token_ids`、`sampling_params`、到达时间、优先级。
- 跟踪 `_output_token_ids` 和 `_all_token_ids`。
- 支持 `append_output_token_ids()` 后同步更新 block hashes。
- 支持事件记录、完成状态、prefix cache 跳过标记。

`kvcore/utils/sampling_params.py` 提供最小 `SamplingParams`：

- 当前核心字段是 `max_tokens`。
- `skip_reading_prefix_cache` 控制请求是否绕过 prefix cache 命中。
- `extra_args` 暂时作为扩展口。

设计限制：

- 不支持 `prompt_embeds`。
- 不支持 pooling。
- 不支持多模态、LoRA、结构化输出。

## Model Loading And Model Classes

`kvcore/model/model_loader` 分三层：

- `ModelLoadConfig`：描述模型路径、revision、load format、device、attention backend。
- `DefaultModelLoader`：读取 HF config，按 `model_type` 创建模型，遍历 safetensors/bin/pt 权重。
- 模型类自己的 `load_weights()`：完成 HF 权重名到本地参数的加载。

当前模型：

- `kvcore/model/models/llama3.py`
- `kvcore/model/models/qwen3.py`
- `kvcore/model/models/mistral.py`

模型结构大致是：

```text
ForCausalLM
  -> Model
    -> embed_tokens
    -> decoder layers
      -> Attention
      -> MLP
    -> norm
  -> lm_head
```

模型 forward 仍是 eager PyTorch 路径，接受：

- `input_ids`
- `positions`
- `attn_metadata`

模型类不再接收 `kv_caches` 参数。和 vLLM 当前主线的分层类似，模型 forward 只接收 tokens/positions/runtime metadata；KV cache 不作为 per-layer 参数从模型入口注入。
KVCore 的差异是：当前只有一个全局共享大 KV tensor，因此 attention backend 通过 `PagedAttentionMetadata.kv_cache_tensor + block_tables + slot_mapping + layer_idx` 定位本层读写位置。

## Attention Boundary

`kvcore/model/layer/attention.py` 是模型层和 backend 之间的边界：

- 模型专属 attention 层负责 Q/K/V projection、RoPE、输出 projection。
- 通用 `Attention` wrapper 负责 q/k/v 形状标准化、backend 调用、输出 reshape。
- Attention 后端后续通过 `attn_metadata.kv_cache_tensor` 和每层 block table
  定位全局物理 KV 块；ModelRunner 不再绑定 per-layer KV tensor。

当前 backend：

- `kvcore/model/attn_backend/base.py` 定义协议。
- `kvcore/model/attn_backend/torch_paged.py` 提供低性能但语义正确的 paged attention reference backend。
- `kvcore/model/attn_backend/triton_paged.py` 提供当前 CUDA paged KV runtime backend。

当前限制：

- `torch_paged` 用于 correctness oracle，不用于性能。
- `triton_paged` 已经支持 paged write 和基于 block table 的 causal attention，但仍需要更大 correctness matrix 和真实模型验证。
- 原始 `torch_sdpa` backend 已删除，避免非 paged 路径掩盖 KV cache 语义问题。

## KV Block Pool

`kvcore/kv/kv_utils.py` 定义底层 block 元数据：

- `KVBlock`
- `BlockHash`
- `BlockHashWithGroupId`
- `FreeKVBlockQueue`
- block hash 计算工具

`kvcore/kv/block_pool.py` 管理全局物理 block 池：

- block 0 是 `null_block`。
- `get_new_blocks()` 从 free queue 分配。
- `free_blocks()` 按 ref count 释放。
- `touch()` 用于 prefix cache 命中时增加 ref count，并从 free queue 移除。
- `cache_full_blocks()` 将完整 block 写入 prefix cache hash map。
- `reset_prefix_cache()` 清空 prefix cache。

当前设计选择：

- 所有层共享一个全局 `BlockPool`。
- `KVBlock.block_id` 是全局物理 page id。
- 每层逻辑 block table 可以引用不同的物理 block id。

## KVManager

`kvcore/kv/kv_manager.py` 是 KV 生命周期入口。

它只做 block 生命周期，不接触 torch tensor，不构造模型输入：

- `get_computed_blocks(request)`
- `can_fit(request, num_new_tokens, new_computed_blocks=None)`
- `allocate_slots(request, num_new_tokens, new_computed_blocks=None)`
- `cache_blocks(request, num_computed_tokens)`
- `free(request)`
- `get_blocks(request_id)`
- `get_block_ids(request_id)`
- `take_new_block_ids()`
- `evict_request_blocks(selections)`

`KVManagerConfig` 描述：

- `num_gpu_blocks`
- `max_model_len`
- `layer_specs`
- `enable_caching`

`KVCacheBlocks` 是 KVManager 和 ModelRunner 之间的只读交接对象：

```text
KVCacheBlocks.blocks[layer_idx][logical_block_idx] -> KVBlock
```

## SingleTypeKVManager

`kvcore/kv/single_type_kv_manager.py` 管每一层的逻辑 block 表。

当前有：

- `SingleTypeKVManager`
- `FullAttentionKVManager`
- `SlidingWindowKVManager`

每层 manager 维护：

- `req_to_blocks: request_id -> list[KVBlock]`
- `num_cached_blocks`
- `permanently_evicted_blocks`
- `new_block_ids`

Full attention 已可运行：

- prefix cache 从左到右查找连续 block hash 命中。
- 不跳过历史 token。

Sliding window 目前只保留扩展点：

- `KVCacheType.SLIDING_WINDOW`
- `sliding_window`
- `get_num_skipped_tokens()`
- cache hit 逻辑暂未实现。

## Permanent Eviction And Compute Sparsity

永久驱逐属于 KVManager：

- 输入是 `LayerBlockSelection(request_id, layer_idx, block_indices)`。
- `block_indices` 是逻辑 block index，不是物理 `block_id`。
- 支持随机、不连续选择。
- 目标逻辑位置替换为 `null_block`。
- 原物理 block 的 ref count 下降。
- 该 request/layer 后续不会重新缓存这些逻辑位置。

计算稀疏属于 ModelRunner：

- `kvcore/model/kv_runtime.py` 定义 `SparseComputePlan`。
- 每次 forward 前临时传入。
- 只影响 ModelRunner 构造 block table 和 slot mapping。
- 不改变 `KVManager.req_to_blocks`。
- 不改变物理 KV tensor。
- 不改变 prefix cache。

两者区别：

```text
永久驱逐：改变 KV 生命周期和逻辑 block table。
计算稀疏：只改变本次 attention metadata。
```

## BlockTable

`kvcore/kv/block_table.py` 对齐 vLLM `v1/worker/block_table.py` 的通用接口，而不是 GPU staged-write/UVA 版本。

核心对象：

- `CpuGpuBuffer`
- `BlockTable`
- `MultiGroupBlockTable`

`BlockTable` 支持：

- `append_row(block_ids, row_idx)`
- `add_row(block_ids, row_idx)`
- `clear_row(row_idx)`
- `move_row(src, tgt)`
- `swap_row(src, tgt)`
- `commit_block_table(num_reqs)`
- `compute_slot_mapping(num_reqs, query_start_loc, positions)`
- `get_device_tensor(num_reqs)`
- `get_cpu_tensor()`
- `get_numpy_array()`

当前不包含 PCP/DCP/context parallel 逻辑。

slot mapping 简化为：

```text
block_index = position // block_size
block_offset = position % block_size
slot_id = block_number * block_size + block_offset
```

`MultiGroupBlockTable` 管多层/多 KV group 的 `BlockTable` 列表。

## ModelRunner Runtime Boundary

`kvcore/model/model_runner.py` 当前有两类职责。

模型生命周期：

- `create_model()`
- `load_model()`

KV runtime：

- `initialize_kv_cache(kv_manager_config)`
- `initialize_kv_cache_tensor(kv_manager_config)`
- `build_block_tables(request_ids, sparse_plan=None)`
- `build_slot_mapping(block_tables)`
- `build_attention_metadata(request_ids, sparse_plan=None)`

KV tensor 首版布局是一块所有层共享的连续大 tensor：

```text
[2, num_gpu_blocks, block_size, num_kv_heads, head_size]
```

这里 `num_gpu_blocks` 是所有层共享的全局物理块总数。层级差异不体现在 tensor 维度上，而体现在 KVManager 的 per-layer block table 和 ModelRunner 构建的 metadata 中。

`kvcore/model/kv_runtime.py` 定义 runner 侧运行时数据：

- `SparseComputePlan`
- `PagedAttentionMetadata`

`ModelRunner` 是唯一把 KVManager 的逻辑 block 状态转换为模型/attention backend 输入的地方。

`ModelRunner.profile_run()` 会生成一次轻量 KV cache profile 结果，`EngineCore` 只保存该结果并把解析后的 block 数交给 `Scheduler/KVManager`：

- 手动设置 `EngineConfig.num_gpu_blocks` 时，runner profile 记录当前配置能支撑的单序列 token 上限。
- 设置 `EngineConfig.profile_kv_cache=True` 或 `num_gpu_blocks=None` 时，runner 会根据当前设备可用内存和单个 KV block 字节数估算 `num_gpu_blocks`。
- 该 profile 遵循 vLLM “runner owns model/device/KV tensor profiling” 的职责边界，但不引入 worker/executor 层。

## Current Execution Flow

### 1. Create Or Load Model

```text
ModelRunner(load_config)
  -> create_model()
     -> DefaultModelLoader.load_config_from_source()
     -> DefaultModelLoader.create_model()

或

ModelRunner(load_config)
  -> load_model()
     -> create model
     -> load checkpoint weights
     -> move to device
     -> eval()
```

### 2. Initialize KV Runtime

```text
KVLayerSpec per layer
  -> KVManagerConfig
  -> Scheduler(KVManagerConfig)
     -> KVManager(config)
  -> ModelRunner.initialize_kv_cache()
     -> initialize_kv_cache_tensor()
     -> shared torch.Tensor KV cache
```

### 3. Add Request

```text
Request(prompt_token_ids, SamplingParams)
  -> KVManager.get_computed_blocks()
  -> KVManager.can_fit()
  -> KVManager.allocate_slots()
     -> SingleTypeKVManager.allocate_new_blocks()
     -> BlockPool.get_new_blocks()
     -> optional cache_blocks()
```

### 4. Build Runtime Inputs

```text
ModelRunner.build_attention_metadata(request_ids, sparse_plan)
  -> build_block_tables()
     -> KVManager.get_blocks(request_id)
     -> skip null blocks
     -> skip SparseComputePlan logical blocks
     -> MultiGroupBlockTable.add_row()
     -> commit_block_table()
  -> build_slot_mapping()
     -> BlockTable.compute_slot_mapping()
  -> PagedAttentionMetadata
```

### 5. Model Forward

```text
model(input_ids, positions, attn_metadata)
  -> model layers
  -> model-specific attention
  -> generic Attention wrapper
  -> backend.forward(query, key, value, attn_metadata, layer_idx)
```

backend 语义分三类：

- `torch_paged`: PyTorch reference backend，按 `slot_mapping` 写共享 KV tensor，并按 block table gather 历史 K/V。
- `triton_paged`: CUDA runtime backend，目标语义应与 `torch_paged` 对齐。

### 6. Finish Request

```text
KVManager.free(request)
  -> each SingleTypeKVManager.free(request_id)
  -> BlockPool.free_blocks()
```

## Module Ownership Summary

| Module | Owns | Does Not Own |
|---|---|---|
| `Request` | request token state | KV allocation |
| `ModelLoader` | model creation and weight loading | runtime KV metadata |
| `ModelRunner` | KV tensors and backend/model metadata | block lifecycle policy |
| `KVManager` | block lifecycle and logical block tables | torch tensors |
| `SingleTypeKVManager` | per-layer request block table | global scheduling |
| `BlockPool` | physical KVBlock allocation/ref count/cache hash | model metadata |
| `BlockTable` | runner-side block table and slot mapping | KVBlock ownership |
| `Attention` | backend dispatch | block allocation |

## Current Limits

- Scheduler-to-runner 最小执行循环已经存在，但还不是完整 vLLM 级别的 serving loop。
- Sliding-window manager is only an interface skeleton.
- `torch_paged` 是 correctness reference，性能很低。
- `triton_paged` 已有 paged KV write/read，但需要更多多请求、GQA、真实模型 logits 对齐测试。
- 还没有 CPU offload、preemption、speculative decoding、distributed executor、完整 prefix-cache benchmark。
