# 模型创建与权重加载设计记录

日期: 2026-04-20 23:59

## problem statement

当前仓库已经有最小的 decoder-only 模型骨架，但还缺少一条完整的模型生命周期前半段：

- 创建模型骨架
- 从 Hugging Face 格式权重加载

并且这条链路需要和 vLLM 的组织方式保持一致的思想：

- model runner 负责模型创建与加载的 orchestration
- model_loader 负责具体权重解析与灌入
- 先不实现 profiling / KV cache 初始化 / 执行循环

## hypothesis

如果把“创建模型”和“加载权重”明确拆成两层：

- `ModelRunner`
- `ModelLoader`

那么后续继续接：

- 显存 profiling
- KV cache 初始化
- 运行时 execute loop

时，边界会更稳定，也更接近 vLLM 的演进方式。

## method

本次新增：

- `kvcore/model/model_runner.py`
- `kvcore/model/model_loader/base.py`
- `kvcore/model/model_loader/hf_loader.py`

### `ModelRunner`

当前只负责最小生命周期前半段：

1. `create_model()`
2. `load_model()`

它不负责：

- profiling
- KV cache 初始化
- 推理执行循环

### `DefaultModelLoader`

当前负责：

1. `AutoConfig.from_pretrained(...)`
2. 根据 `model_type` 选择本地模型类
3. 创建空模型骨架
4. 按 `load_format` 解析 Hugging Face 权重文件
5. 通过权重迭代器逐个产出 tensor
6. 调用模型自己的 `load_weights(...)`

当前结构上已经参考了 vLLM `DefaultModelLoader` 的几个关键分层：

- `Source`
- `_prepare_weights(...)`
- `_get_weights_iterator(...)`
- `get_all_weights(...)`
- `load_weights(...)`

当前支持的权重格式：

- `model.safetensors`
- `model.safetensors.index.json`
- `pytorch_model.bin`
- `pytorch_model.bin.index.json`
- `load_format=auto/hf/safetensors/mistral/pt`

同时保留了：

- `HuggingFaceModelLoader`

作为兼容别名，避免现有调用点被打断。

### 模型注册

当前 `model_type -> model class` 映射为：

- `llama -> Llama3ForCausalLM`
- `mistral -> MistralForCausalLM`
- `qwen3 -> Qwen3ForCausalLM`

## result

当前生命周期已变为：

```text
ModelRunner.create_model()
    -> AutoConfig
    -> create empty nn.Module

ModelRunner.load_model()
    -> DefaultModelLoader.load_model()
    -> load weights into model
```

这和 vLLM 的“先创建骨架，再加载权重，再继续 runtime 初始化”的思路保持一致。

## experiment

测试文件：

- `tests/test_model_loading.py`

覆盖内容：

1. 本地 HF config 创建 Llama3 模型骨架
2. 从 `pytorch_model.bin` 加载 Llama3 权重
3. 从 `model.safetensors` 加载 Qwen3 权重
4. `ModelRunner.create_model()` 与 `load_model()` 的顺序行为
5. `DefaultModelLoader` 的默认 `hf` 加载格式

测试方法：

- 使用临时目录写出本地 HF 风格目录
- `config.save_pretrained(...)`
- 写出 `pytorch_model.bin` 或 `model.safetensors`
- 再通过 loader / runner 重新加载并比对 state dict

## limits

当前仍未实现：

- 远端真实大模型下载的集成测试
- `revision` / `trust_remote_code` 的复杂路径验证
- profiling
- KV cache 初始化
- 运行时分配与执行循环
- 分布式 / 量化 / shard 加载

## conclusion

当前已经把 vLLM 风格的“模型骨架创建”和“权重加载”拆开实现出来了：

- `ModelRunner` 负责编排
- `DefaultModelLoader` 负责默认 Hugging Face 权重加载

这为下一阶段继续接：

- profiling
- KV cache 初始化
- engine / runner 真实执行路径

提供了一个清晰且可测试的基础。
