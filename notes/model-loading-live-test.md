# Llama-3.1-8B-Instruct 实机加载与前向测试记录

日期: 2026-04-21 00:15

## problem statement

需要使用当前仓库代码，对本地模型目录：

- `/Tan/model/Llama-3.1-8B-Instruct/`

执行一条完整的前半生命周期测试：

1. 模型初始化
2. 权重加载
3. fake 数据 forward

并输出每一步的测试结果。

## environment

- Python 环境：项目 `.venv`
- 模型路径：`/Tan/model/Llama-3.1-8B-Instruct/`
- 可用 GPU：
  - `cuda:0` = `NVIDIA L40S`
  - `cuda:1` = `NVIDIA GeForce RTX 3090`
  - `cuda:2` = `NVIDIA RTX A6000`
- 本次实际选择设备：`cuda:0`
- 配置读取到的模型 dtype：`torch.bfloat16`

## pre-check

模型目录检查结果：

- `exists = True`
- `config.json = True`
- `model.safetensors.index.json = True`
- `safetensors shard_count = 4`

## note

在执行真实测试前，发现当前手动修改后的代码里存在一个导入链断点：

- `kvcore/model/model_loader/__init__.py`
  - 仍在导出 `HuggingFaceModelLoader`
- `kvcore/model/model_loader/default_loader.py`
  - 已没有这个名字

因此先做了一个**最小兼容修补**：

- 在 `default_loader.py` 中添加
  - `class HuggingFaceModelLoader(DefaultModelLoader): ...`

该修补不改变当前默认执行逻辑，只是恢复导入兼容性，使测试链路能够继续执行。

## step-by-step result

### step 1: check model path

- `exists = True`
- `config.json = True`
- `safetensors index = True`
- `shard_count = 4`

结论：

- 模型目录完整，可继续执行。

### step 2: load huggingface config

结果：

- `model_type = llama`
- `hidden_size = 4096`
- `num_hidden_layers = 32`
- `num_attention_heads = 32`
- `num_key_value_heads = 8`
- `vocab_size = 128256`
- `torch_dtype = torch.bfloat16`
- `time_sec = 0.0`

结论：

- `AutoConfig.from_pretrained(...)` 正常
- 当前目录确认为 Llama 系列模型
- 8B 配置读取正确

### step 3: choose device and dtype

结果：

- `device = cuda:0`
- `chosen_default_dtype = torch.bfloat16`
- `pre_load_gpu_mem_free_gib = 33.9`
- `pre_load_gpu_mem_total_gib = 44.31`

结论：

- 选用了空闲显存最多的 GPU
- 用 `bfloat16` 创建模型骨架，符合配置里的原始 dtype

### step 4: create empty model skeleton via ModelRunner.create_model()

结果：

- `model_class = Llama3ForCausalLM`
- `param_count = 8030261248`
- `first_param_dtype = torch.bfloat16`
- `first_param_device = cpu`
- `time_sec = 54.48`

结论：

- 当前 `ModelRunner.create_model()` 可成功创建 8B 空骨架
- 参数规模约 `8.03B`
- 初始化阶段参数在 CPU 上分配，dtype 为 `bfloat16`

### step 5: load weights via ModelRunner.load_model()

结果：

- `loaded_model_class = Llama3ForCausalLM`
- `loaded_first_param_dtype = torch.bfloat16`
- `loaded_first_param_device = cuda:0`
- `time_sec = 71.74`
- `post_load_gpu_mem_free_gib = 20.32`
- `post_load_gpu_mem_total_gib = 44.31`

结论：

- 当前 `ModelRunner.load_model()` 可以成功完成：
  - 配置加载
  - 模型骨架创建
  - safetensors shard 权重加载
  - 模型迁移到目标 GPU
- 权重加载后显存占用符合 8B bf16 规模预期

### step 6: fake forward test

fake 输入：

- `batch_size = 1`
- `seq_len = 4`

结果：

- `input_ids_shape = (1, 4)`
- `positions_shape = (1, 4)`
- `hidden_states_shape = (1, 4, 4096)`
- `hidden_states_dtype = torch.bfloat16`
- `logits_shape = (1, 4, 128256)`
- `logits_dtype = torch.bfloat16`
- `logits_finite = True`
- `time_sec = 1.45`

结论：

- 当前模型在真实 8B 权重下可以完成 fake forward
- `forward()` 与 `compute_logits()` 链路都已打通
- 输出张量数值有效，没有出现 `NaN/Inf`

### step 7: post-forward gpu memory

结果：

- `gpu_mem_free_gib = 20.03`
- `gpu_mem_total_gib = 44.31`

结论：

- 小 batch 小 seq fake forward 后显存波动正常

## final conclusion

本次对 `/Tan/model/Llama-3.1-8B-Instruct/` 的真实测试结果为：

- 模型目录检查：通过
- HF config 加载：通过
- 空模型骨架创建：通过
- 权重加载：通过
- fake forward：通过
- logits 计算：通过

最终状态：

- `FINAL: success`

## limits

本次测试仍未覆盖：

- tokenizer 编码真实文本输入
- 长序列 prefill
- decode 阶段缓存复用
- profiling
- KV cache 初始化
- engine / scheduler / request 路径集成

