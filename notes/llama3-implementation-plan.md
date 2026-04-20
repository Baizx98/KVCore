# Llama3 第一阶段实现规划

日期: 2026-04-20

## 1. 目标

### problem statement
当前仓库已经有 `kvcore/model/models/llama3.py`、`kvcore/model/layer/`、`kvcore/model/attn_backend/` 的目录骨架，但尚未实现可用的 Llama3 推理模型结构。目标是先按 vLLM 的模块组织方式实现一版 **单卡、decoder-only、inference-only** 的 Llama3 架构，并把 attention 计算外包给 attn backend。

### hypothesis
如果第一版先把模型结构、参数命名、层间接口、KV/attention 元数据边界设计稳定，那么后续接入真正的 paged attention backend、KV manager、权重加载和调度器时，返工会显著减少。

### method
严格借鉴 vLLM 的模块拆分方式，但主动删除当前项目暂不需要的复杂度：

- 不支持 TP、PP、LoRA、量化、Eagle、编译装饰器。
- 不在模型内部实现 attention kernel。
- 不在第一版里耦合完整 `KVManager` 逻辑，只预留 metadata / cache 接口。
- 尽量保持 HF / vLLM 风格的参数命名，便于后续兼容权重加载。

### expected result
得到一套可以稳定演进的最小 Llama3 模型骨架：

- `embed_tokens -> N x decoder layer -> norm -> lm_head`
- 每层包含 `RMSNorm -> Attention -> RMSNorm -> MLP`
- attention 层只负责 `q/k/v/o`、RoPE、调用 backend
- `llama3.py` 暴露 `Llama3Model` / `Llama3ForCausalLM`

## 2. 与 vLLM 对齐的核心点

以下部分建议尽量对齐 vLLM / HF 命名与结构：

- `LlamaMLP`
  - `gate_proj`
  - `up_proj`
  - `down_proj`
  - 激活函数固定 `silu`
- `LlamaAttention`
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `num_attention_heads`
  - `num_key_value_heads`
  - `head_dim`
  - `scaling`
  - `rotary embedding`
- `LlamaDecoderLayer`
  - `input_layernorm`
  - `self_attn`
  - `post_attention_layernorm`
  - `mlp`
- `Llama3Model`
  - `embed_tokens`
  - `layers`
  - `norm`
- `Llama3ForCausalLM`
  - `model`
  - `lm_head`
  - `forward`
  - `compute_logits`

这样做的直接收益是：

- 后续支持 HF 权重加载时命名天然兼容。
- 和 vLLM / transformers 对照阅读成本低。
- 以后若扩展到 Mistral / Qwen3，可以复用大部分层实现。

## 3. 第一版建议新增的文件

建议第一阶段至少补齐以下文件：

- `kvcore/model/layer/rmsnorm.py`
- `kvcore/model/layer/rotary_embedding.py`
- `kvcore/model/layer/activation.py`
- `kvcore/model/layer/linear.py`
- `kvcore/model/layer/attention.py`
- `kvcore/model/layer/mlp.py`
- `kvcore/model/layer/decoder.py`
- `kvcore/model/models/llama3.py`

如果你更希望保持文件更少，也可以压缩为：

- `kvcore/model/layer/llama.py`
- `kvcore/model/models/llama3.py`

但从后续扩展到 Mistral / Qwen3 的角度看，我更推荐前一种“按基础层拆开”的组织方式。

## 4. 各层的职责边界

### 4.1 `activation.py`

建议提供：

- `SiluAndMul`

职责：

- 输入形状为 `[..., 2 * intermediate_size]`
- 切成两半后执行 `silu(gate) * up`
- 对齐 vLLM 的 fused-MLP 表达方式，但内部先用普通 PyTorch 实现即可

### 4.2 `linear.py`

建议第一版只提供最薄的封装：

- `ColumnLinear`
- `RowLinear`
- `MergedColumnLinear`
- `QKVLinear`

职责：

- 本质上还是 `nn.Linear`
- 先不做 TP shard
- 保留和 vLLM 类似的构造接口，减少未来替换成本

建议：

- `MergedColumnLinear` 输出拼接后的张量，供 MLP 中 `gate/up` 共用一次 GEMM
- `QKVLinear` 一次投影出拼接后的 `qkv`

### 4.3 `rmsnorm.py`

建议提供：

- `RMSNorm`

职责：

- 对齐 LLaMA/Llama3 的 RMSNorm 行为
- 支持标准 `forward(x) -> y`

第一版先不要做 vLLM 里那种 `(hidden_states, residual)` 的 fused residual 接口，因为你当前框架还没有 pipeline / compiled graph 的约束，先保持直观实现更合适。

### 4.4 `rotary_embedding.py`

建议提供：

- `LlamaRotaryEmbedding`
- `apply_rotary_pos_emb`

职责：

- 从 config 读取 `rope_parameters`、`max_position_embeddings`、`head_dim`
- 给定 `positions` 生成 `cos/sin`
- 对 `q/k` 应用 RoPE

建议的输入输出：

- 输入: `positions`, `q`, `k`
- 输出: `q_rot`, `k_rot`

### 4.5 `attention.py`

建议包含两部分：

- `AttentionBackend` 抽象接口
- `LlamaAttention`

其中 `AttentionBackend` 第一版只需要定义接口，不需要真正高性能实现。

建议 backend 接口类似：

```python
class AttentionBackend(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_metadata: object | None = None,
        kv_cache: object | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor: ...
```

`LlamaAttention` 的职责：

- `qkv_proj`
- reshape 成 multi-head 视图
- 对 `q/k` 施加 rotary
- 调用 backend 完成 attention
- `o_proj` 投回 hidden size

第一版不要做的事：

- 不在这里实现 causal mask 细节
- 不在这里管理 block table
- 不在这里自己维护 paged KV 物理布局

这些应由 backend + 后续 attn metadata 承担。

### 4.6 `mlp.py`

建议提供：

- `LlamaMLP`

结构：

- `gate_up_proj = MergedColumnLinear(hidden, [intermediate, intermediate])`
- `act = SiluAndMul()`
- `down_proj = RowLinear(intermediate, hidden)`

### 4.7 `decoder.py`

建议提供：

- `LlamaDecoderLayer`

forward 建议保持标准 residual 形式：

```python
residual = hidden_states
hidden_states = input_layernorm(hidden_states)
hidden_states = self_attn(...)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = residual + hidden_states
```

这比 vLLM 的 fused residual 接口更适合当前最小框架，逻辑更清楚。

## 5. `llama3.py` 里的模型结构

## 5.1 `Llama3Model`

建议成员：

- `config`
- `embed_tokens`
- `layers`
- `norm`
- `rotary_emb`
- `attn_backend`

这里有一个关键设计决定：

### 建议把 `rotary_emb` 放在 `Attention` 内，而不是 `Model` 顶层

原因：

- 更接近 vLLM 的写法
- 每层自包含，后续做 layer-wise runner 更自然
- 当前仓库强调 layer-by-layer execution，这样接口更直观

因此 `Llama3Model` 主要职责是：

- embedding
- 逐层调用 decoder layer
- final norm

建议 `forward` 签名先定为：

```python
def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    *,
    inputs_embeds: torch.Tensor | None = None,
    attn_metadata: object | None = None,
    kv_caches: list[object] | None = None,
) -> torch.Tensor:
    ...
```

说明：

- `positions` 单独传入，方便以后 prefill / decode 统一。
- `attn_metadata` 作为 opaque object 传给 backend。
- `kv_caches` 用 `list[object]` 先占位，每层一个 cache handle，后续再替换成正式类型。

## 5.2 `Llama3ForCausalLM`

建议成员：

- `model`
- `lm_head`

建议接口：

```python
def forward(...) -> torch.Tensor
def compute_logits(hidden_states: torch.Tensor) -> torch.Tensor
```

第一版先不要在这里引入 sampler / logits processor。

原因：

- 你当前仓库还没有 sampling 子系统
- 直接输出 logits 更适合作为模型子模块的第一步

## 6. attention backend 的第一版边界

这是这次规划最关键的一点。

### problem statement
你明确要求 attention 通过 `attn backend` 完成，而不是在 Llama 模型里直接写 attention 计算。

### 方法建议
模型层只做三件事：

1. 线性投影出 `q/k/v`
2. 应用 rotary
3. 调用 `backend.forward(...)`

backend 负责：

- causal/prefill/decode 路径区分
- 是否读写 KV cache
- 是否使用 paged/blocked layout
- 未来切换到 Triton / Flash / 自定义 kernel

### 第一版最小实现建议

即使你说“具体实现之后再说”，我仍建议先提供一个能跑通形状的占位 backend，例如：

- `DummyAttentionBackend`
- `TorchSDPAAttentionBackend`

二选一均可。

如果你想最小化工作量，可以只写：

- 抽象基类 / protocol
- 一个 `NotImplementedError` 占位 backend

但我更推荐至少补一个最小 eager backend，原因是：

- 后面验证 shape / 权重加载更容易
- 单测更容易写
- decoder layer 链路可以尽早打通

## 7. 配置与权重兼容建议

### 配置来源

建议直接依赖 HF `LlamaConfig`，不要自定义另一套 config 类。

核心字段至少使用：

- `vocab_size`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `hidden_act`
- `max_position_embeddings`
- `rms_norm_eps`
- `attention_bias`
- `mlp_bias`
- `head_dim`
- `rope_parameters`
- `tie_word_embeddings`

### 权重命名

第一版虽然可以不马上实现完整 `load_weights`，但模块命名应兼容以下 HF 名称：

- `model.embed_tokens.weight`
- `model.layers.{i}.self_attn.q_proj.weight`
- `model.layers.{i}.self_attn.k_proj.weight`
- `model.layers.{i}.self_attn.v_proj.weight`
- `model.layers.{i}.self_attn.o_proj.weight`
- `model.layers.{i}.mlp.gate_proj.weight`
- `model.layers.{i}.mlp.up_proj.weight`
- `model.layers.{i}.mlp.down_proj.weight`
- `model.layers.{i}.input_layernorm.weight`
- `model.layers.{i}.post_attention_layernorm.weight`
- `model.norm.weight`
- `lm_head.weight`

### 建议

即使内部用了 `QKVLinear` / `MergedColumnLinear`，也要提前决定：

- 是保留拆开的 `q_proj/k_proj/v_proj` 参数
- 还是保留 packed 参数并写映射 loader

我的建议是：

### 第一阶段保留拆开的 `q_proj/k_proj/v_proj` 与 `gate_proj/up_proj`

原因：

- 最容易直接加载 HF 权重
- 最容易 debug
- 更适合你现在这个“最小研究框架”

### 第二阶段再替换为 packed projection

原因：

- packed 更贴近 vLLM 的执行优化
- 但它会把权重加载、参数命名、debug 难度一起抬高

这是一处我建议和 vLLM 做“策略性不完全一致”的地方。

## 8. 我建议的第一阶段实现范围

### 必做

- `RMSNorm`
- `RoPE`
- `LlamaAttention` 结构
- `LlamaMLP`
- `LlamaDecoderLayer`
- `Llama3Model`
- `Llama3ForCausalLM`
- attention backend 抽象接口
- 一个最小 backend 占位实现

### 暂不做

- TP / PP
- paged attention kernel
- block table 接入
- quantization
- fused residual norm
- logits processor / sampling
- 多模型共享基类抽象
- 完整权重加载器

## 9. 风险与取舍

### 风险 1
如果第一版就照抄 vLLM 的 packed linear、fused residual、intermediate tensors 设计，当前仓库会引入很多暂时用不到的复杂度。

影响：

- 代码可读性差
- 后续 debug 困难
- 与你现在“精简版推理框架”的目标不一致

### 风险 2
如果 attention backend 接口现在定义得太窄，后面接 paged KV 时可能要改动 decoder layer 签名。

缓解：

- 现在就保留 `attn_metadata` 与 `kv_cache/kv_caches` 占位
- layer 级别显式传 `layer_idx`

### 风险 3
如果一开始就强依赖完整 `KVManager`，模型模块会被 runtime 细节污染。

缓解：

- 让模型只依赖 opaque metadata / cache handle
- backend 适配 runtime

## 10. 最推荐的落地顺序

1. 先实现基础层
   - `RMSNorm`
   - `RoPE`
   - `SiluAndMul`
   - 基础 linear wrapper
2. 再实现 layer
   - `LlamaMLP`
   - `LlamaAttention`
   - `LlamaDecoderLayer`
3. 再实现模型
   - `Llama3Model`
   - `Llama3ForCausalLM`
4. 最后补最小 backend
   - 先打通 forward 链路
   - 后续再接真正 KV / paged backend

## 11. 我建议你确认的两个关键点

### 关键点 A
第一版是否接受“参数命名对齐 HF，内部实现不强行 packed 化”。

我的建议：接受。

原因：

- 更稳
- 更适合研究原型
- 后续要向 packed 重构也不难

### 关键点 B
第一版是否允许提供一个最小可运行 backend，而不是只有纯接口。

我的建议：允许。

原因：

- 可以尽早做 shape 验证
- 能更快发现 forward 签名设计问题

## 12. 最终结论

### conclusion
建议把 Llama3 第一版实现为一个“结构严格参考 vLLM、执行路径适配最小单卡框架”的版本：

- 架构层面贴近 vLLM / HF
- 接口层面为后续 attn backend / KV cache 演化留口
- 实现层面避免一次引入 TP/PP/fused/packed 等额外复杂度

如果按这个方案推进，后续编码阶段我会优先保证：

- 文件组织清晰
- 接口稳定
- 命名兼容 HF
- attention backend 边界干净
- 便于后续继续接 paged KV 和 scheduler

