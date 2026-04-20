# Decoder-Only 模型当前流程测试记录

日期: 2026-04-20

## problem statement

需要在当前仓库和当前 `.venv` 环境下，验证当前已经实现的 decoder-only 模型骨架是否已经打通最小模型流程，而不是只停留在静态检查。

## hypothesis

如果当前实现的模型骨架、attention backend 接口和 KV cache 占位接口没有明显错误，那么以下最小链路应当可以稳定通过：

- `Attention` 抽象层调用链
- batched `forward`
- flat token `forward`
- `compute_logits`
- `tie_word_embeddings`
- `kv_cache.update` 调用传播

## method

测试文件：

- `tests/test_llama3_flow.py`

测试环境：

- 使用当前项目 `.venv`
- `import torch` 约 1.31s
- `from transformers import LlamaConfig` 约 31.16s

执行命令：

```bash
PYTHONPATH=/home10T/bzx/workspace/KVCore .venv/bin/ruff check tests/test_llama3_flow.py
PYTHONPATH=/home10T/bzx/workspace/KVCore .venv/bin/pytest tests/test_llama3_flow.py -q
```

## experiment

当前测试覆盖了 8 条流程：

1. `Attention` 抽象层 batched 调用
2. `Attention.bind_kv_cache`
3. `Llama3` batched forward + logits shape
4. `Llama3` flat token forward + logits shape
5. `Llama3` tied embedding 权重别名
6. `Llama3` `kv_caches` 逐层传递以及 `kv_cache.update` 调用
7. `Qwen3` batched forward + logits shape
8. `Mistral` batched forward + logits shape

测试配置：

- `vocab_size=128`
- `hidden_size=64`
- `intermediate_size=128`
- `num_hidden_layers=2`
- `num_attention_heads=4`
- `num_key_value_heads=2`
- `max_position_embeddings=128`
- 固定 `torch.manual_seed(0)`

## result

测试结果：

- `ruff check`: 通过
- `pytest`: `8 passed in 3.11s`

确认通过的行为：

- 模型可以处理 `[batch, seq]` 输入
- 模型可以处理扁平 `[num_tokens]` 输入
- `compute_logits` 输出形状正确
- `tie_word_embeddings=True` 时 `lm_head.weight` 与 `embed_tokens.weight` 共享参数
- `attn_metadata` 和逐层 `kv_cache` 能透传到 attention backend
- backend 会按层调用 `kv_cache.update`

## limits

当前测试仍然没有覆盖：

- HuggingFace 实际权重加载
- 真正的 paged KV / block table
- prefill / decode 分流逻辑
- attention mask / sliding window 的复杂形态
- 长序列、数值稳定性和性能
- 真实 scheduler / engine 集成

## conclusion

当前 `Llama3 / Qwen3 / Mistral` 第一版已经通过最小流程测试，并且 attention 分层已经调整为：

- 通用 `Attention` 抽象保留在 `kvcore/model/layer/attention.py`
- 模型专属的 `*MLP / *Attention / *DecoderLayer` 回收到各自 model 文件

现在可以作为后续继续接：

- 正式 `attn_metadata` 类型
- KV cache 结构
- HF 权重加载
- engine / scheduler 集成

的一条可用基线。
