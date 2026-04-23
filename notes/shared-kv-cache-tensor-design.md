# 共享 KV Cache Tensor 设计

日期：2026-04-23 21:46:15 CST

## 问题

之前 `ModelRunner` 为每一层单独创建一个 KV cache tensor：

```text
layer_0 -> [2, num_blocks, block_size, num_kv_heads, head_size]
layer_1 -> [2, num_blocks, block_size, num_kv_heads, head_size]
...
```

这不符合当前设计目标。你希望所有层的 KV cache 在底层共享一个大的物理 tensor，而不是多次独立分配。

## 设计结论

首版采用一个统一 contiguous tensor，不带 layer 维：

```text
[2, num_gpu_blocks, block_size, num_kv_heads, head_size]
```

其中：

- 第 0 维是 key/value。
- 第 1 维是全局物理 block id。
- 后三维是 block 内 token、KV head、head dim。

`num_gpu_blocks` 的含义是所有层共享的全局物理块总数。每层不会在物理 tensor 上拥有独立维度；某一层的某个逻辑块只通过 `KVManager` 中该层的 block id 映射到这个全局 tensor。

Attention 计算时应接收：

- 全局 `kv_cache_tensor`
- 当前层的 `layer_idx`
- 当前层的 block table / slot mapping / active block ids

后端根据 block id 直接定位：

```text
kv_cache_tensor[:, block_id, :, :, :]
```

## 和 vLLM 的关系

vLLM 的 GPUModelRunner 负责创建 KV tensor，attention 后端再结合 metadata 使用这些 tensor。KVCore 省去了 executor/worker 后，保留这个核心思想，但进一步简化为一个全局物理 KV tensor；层级差异只存在于 KVManager 的逻辑块表和 ModelRunner 构建的 metadata 中。

参考：

- vLLM GPUModelRunner: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu_model_runner.py
- vLLM Attention layer: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/attention.py

## 当前假设

首版要求所有层的 KV cache spec 一致：

- `block_size` 一致
- `num_kv_heads` 一致
- `head_size` 一致
- `dtype` 一致

如果未来支持混合结构，例如 sliding window、不同 KV head 数、不同 head dim，则需要显式 layout metadata 或为不同 spec 建立不同的共享 tensor group。

## 预期收益

动机：减少底层内存对象数量，让 KV cache layout 更接近统一 block pool 的心智模型。

预期收益：

- 更容易后续实现跨层统一管理、统计、压缩或迁移。
- 更容易在后端 kernel 中传入一个 base pointer 和全局 block id。
- 保持 ModelRunner 负责物理 tensor，Scheduler/KVManager 负责逻辑 block 生命周期。

## 风险

- 当前实现只支持 uniform layer spec。
- 后续 paged attention backend 需要明确使用全局 block id 的约定。
- 如果不同层未来需要不同 page size，必须引入更复杂的 layout descriptor。

## 最小验证

- 检查 `ModelRunner.kv_cache_tensor.shape` 是 `[2, num_gpu_blocks, ...]`。
- 检查 attention layer 不再绑定 per-layer KV tensor。
- 检查 `KVForwardMetadata.kv_cache_tensor` 指向全局 tensor。
- 检查 Scheduler 仍然只拥有 `KVManager`，不接触物理 tensor。
