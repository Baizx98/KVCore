# Attention 与 vLLM 实现差异分析

日期：2026-04-25

## 结论

当前 KVCore 的 `Attention.forward` 已经在最关键的数据形态上接近 vLLM：模型侧传 flattened token 维度的 Q/K/V，Attention wrapper 内部 view 成 `[num_tokens, heads, head_dim]`，从 forward context 读取 attention metadata，并把 output buffer 交给 backend 写入。

主要差异不在模型侧 shape，而在 KV cache 所有权、backend 调用机制、quant/cache-scale、KV sharing、以及算子封装层级。

## 已对齐的部分

1. forward 参数形态基本一致：
   - vLLM：`forward(query, key, value, output_shape=None)`
   - KVCore：`forward(query, key, value, *, output_shape=None)`

2. output shape 推导一致：
   - 默认按 `query.shape[0]` 推导 `num_tokens`
   - 默认输出为 `[num_tokens, num_heads * head_size_v]`

3. Q/K/V reshape 位置一致：
   - 都在 wrapper 层把 Q/K/V view 成 `[num_tokens, heads, head_dim]`
   - 这样模型实现不需要恢复 batch/seq 维度

4. attention metadata 注入方式一致：
   - 都由 model runner 通过 forward context 注入
   - attention 计算时再从 context 读取 metadata

5. output buffer 思路一致：
   - wrapper 预分配 output
   - backend 写入 rank-3 output view
   - wrapper 最后 reshape 回模型期望输出形状

## 关键差异

### 1. KV cache 所有权不同

vLLM 文档语义是 KV cache 存在 `Attention` 类内部，通过 `self.kv_cache` 访问。但新版本 vLLM 的实际路径已经把很多工作统一到 `unified_kv_cache_update` / `unified_attention_with_output` 中，layer name 用来索引对应 cache。

KVCore 当前不是每个 Attention 持有自己的 `kv_cache`，而是通过 `PagedAttentionMetadata.kv_cache_tensor` 访问一个全局共享大 tensor。层差异由 `layer_idx`、`block_tables[layer_idx]` 和 `slot_mapping[layer_idx]` 表达。

这是刻意保留的架构差异：KVCore 更强调单个共享物理 KV tensor，减少 per-layer cache 对象。

### 2. KV update 与 attention 计算的分层不同

vLLM 在 wrapper 里根据 backend 能力决定是否先调用 `unified_kv_cache_update`，然后调用 `unified_attention_with_output`。也就是说，KV 写入和 attention 计算可以是两个统一算子，也可以由 backend 声明自己是否包含 KV update。

KVCore 当前把 KV 写入放在 backend 内部：

- `TorchPagedAttentionBackend.forward()` 内先 `_paged_write()`，再 gather 历史 KV 做 attention。
- `TritonPagedAttentionBackend.forward()` 内先 `_paged_write()` kernel，再 `_paged_attention()` kernel。

因此 KVCore 的 wrapper 更轻，但 backend 职责更重；vLLM 的 wrapper/ops 层更统一。

### 3. backend 调用机制不同

vLLM 通过 layer name 编码后调用统一 custom op：

- direct call: `unified_attention_with_output(...)`
- torch op: `torch.ops.vllm.unified_attention_with_output(...)`

KVCore 仍是 Python protocol：

- `backend_impl.forward(query, key, value, ..., attn_metadata, layer_idx, output)`

这符合当前 Python-first 的最小架构，但还没有 vLLM 那种统一 op registry、CUDA graph 友好路径、backend capability flag。

### 4. KV sharing 支持不同

vLLM forward 中有 `kv_sharing_target_layer_name` 判断，用来支持复用其他层的 KV cache，必要时跳过本层 KV update。

KVCore 当前没有 layer-to-layer KV sharing。每层都通过自己的 `layer_idx` 写入和读取全局 tensor 中对应逻辑层的 block table/slot mapping。

### 5. quant/cache scale 路径缺失

vLLM Attention 有：

- `calculate_kv_scales`
- `maybe_calc_kv_scales`
- `query_quant`
- `kv_cache_dtype`
- `supports_quant_query_input`

KVCore 当前没有这些路径。当前默认是 fp16/fp32/bf16 常规 tensor correctness 路线，尚未支持 FP8/NVFP4 KV cache 或 query quant。

### 6. key/value None 支持不同

vLLM 支持 `key is None` / `value is None`，主要服务于 KV sharing 或一些特殊 backend。

KVCore 当前 `Attention.forward` 会直接对 key/value 做 `.view(...)`，因此要求 key/value 非空。这对当前 decoder-only paged attention 主路径是合理的，但比 vLLM 窄。

### 7. output_shape 检查更严格

KVCore 显式检查 `output_shape[-1] == num_heads * head_size_v`。vLLM 只取 `hidden_size = output_shape[-1]`，更信任模型/后端组合。

这使 KVCore 更早暴露 shape 错误，但对 MLA 这类 output hidden size 不等于 query hidden size 的未来 backend 可能需要放宽。

### 8. sliding window 只保留字段

vLLM 会在 attention backend/cache metadata 中完整处理 sliding attention、hybrid KV 等复杂策略。

KVCore 当前 `Attention` 有 `sliding_window` 字段，但 paged backend 未真正消费 sliding-window metadata；当前仍是 full causal attention。

## 对当前实现的判断

当前 KVCore 和 vLLM 的核心运行态边界已经一致：

```text
Model-specific attention:
  hidden_states -> q/k/v projection -> RoPE -> Attention(q, k, v)

Generic Attention:
  q/k/v flat hidden -> [num_tokens, heads, dim]
  get_forward_context().attn_metadata
  backend.forward(..., output=output_view)

Backend:
  write K/V to paged cache
  read block table + slot mapping
  compute attention
```

真正未对齐的是 vLLM 更成熟的工程能力：统一 custom op、backend capability、quantized KV、KV sharing、CUDA graph 细节、混合 attention 类型。

## 后续建议

短期不建议继续照搬 vLLM 的 `unified_attention_with_output` 形态，因为 KVCore 当前还没有 custom op registry 和多 backend capability 系统。更合适的下一步是：

1. 保持当前 wrapper 接口不变。
2. 在 backend protocol 中逐步增加 capability 字段，例如是否包含 KV update、是否支持 key/value None、是否支持 output buffer。
3. 等 Triton backend 稳定后，再考虑把 paged write 和 paged attention 合并或抽象成更接近 vLLM 的 unified op。
4. 若未来支持 MLA/不同 value head dim，需要重新评估 `output_shape[-1]` 的严格检查。
