# Attention 抽象层设计记录

日期: 2026-04-20

## problem statement

当前仓库原本只有模型专属的 `LlamaAttention`，它同时负责：

- q/k/v 投影
- RoPE
- backend 调用
- KV cache 传递
- attention 输出 reshape

这对最小原型是够用的，但和 vLLM 的分层思路相比，模型层与执行层的边界还不够清晰。

## hypothesis

如果引入一个通用 `Attention` 封装层，让它专门负责：

- backend 选择与调用
- 可绑定的 `kv_cache`
- `attn_metadata` 透传
- q/k/v 到输出张量的标准化接口

那么后续接入：

- paged KV
- block table / slot mapping
- 多种 attention backend
- 其它 decoder-only 模型

时会更稳，也更接近 vLLM 的演进方向。

## method

本次采用“保留最小复杂度”的 vLLM 风格抽象：

### 新增通用层

- `kvcore/model/layer/attention.py::Attention`

它负责：

- 保存 `num_heads / num_kv_heads / head_size / scale`
- 持有 backend 实例
- 提供 `bind_kv_cache`
- 在 `forward` 中统一调用 backend
- 恢复输出张量形状

### 保留模型专属层

- `kvcore/model/layer/attention.py::LlamaAttention`

它现在只负责：

- q/k/v/o 投影
- rotary embedding
- 调用通用 `Attention`

### 保留简化点

与 vLLM 相比，本次故意不引入：

- forward context
- 自定义 op
- kv cache spec
- quantization
- custom opaque op / torch.compile 适配
- backend registry/selector 的复杂层次

原因是当前 KVCore 仍处于最小研究骨架阶段，先把“抽象边界”搭对，比一次搬入全部工程复杂度更重要。

## result

当前分层已经变为：

- `LlamaAttention`
  - 负责模型结构语义
- `Attention`
  - 负责执行抽象层
- `TorchSDPAAttentionBackend`
  - 负责具体后端实现

即：

`projection + rope -> Attention wrapper -> backend impl -> output reshape -> o_proj`

## gain

预期收益：

- 更贴近 vLLM 的 attention 分层
- 后续 Mistral/Qwen3 可以复用同一个 `Attention`
- `kv_cache` 与 `attn_metadata` 的边界更稳定
- 更适合后面把 slot mapping / paged KV 逐步下沉到抽象层

## risk

当前风险：

- 现在的 `Attention` 仍是简化版，还没有 vLLM 那种上下文注册机制
- backend 协议还比较薄，后续如果接入更多 backend，接口可能再扩
- 当前默认仍假设 decoder causal attention

## conclusion

这次重构的目标不是“复刻 vLLM 全部 attention 工程”，而是先把它最关键的一层设计思想拿过来：

- 模型层描述结构
- 抽象层描述 attention 调用语义
- backend 层描述具体执行

这对 KVCore 后续继续向 paged KV / runtime-managed attention 演进是合适的第一步。

