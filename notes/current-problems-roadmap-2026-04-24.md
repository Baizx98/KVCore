2026-04-24 15:10 CST

# KVCore 当前问题与下一步开发规划

## 0. 检查范围

本次检查基于当前仓库文档、核心代码和测试结果，重点关注 vLLM 风格推理框架的最小闭环：

- 文档：`README.md`、`notes/current-architecture-flow.md`、`notes/kvmanager-e2e-readiness-check.md`、`notes/tokenizer-dataclass-e2e-review-2026-04-24.md`
- 代码：`EngineCore`、`LLMEngine`、`Scheduler`、`KVManager`、`ModelRunner`、`TorchPagedAttentionBackend`、`TritonPagedAttentionBackend`
- 测试：`tests/test_llm_engine.py`、`tests/test_scheduler.py`、`tests/test_model_runner_kv_boundary.py`、`tests/test_triton_paged_attention.py`、全量 pytest

验证结果：

```text
PYTHONPATH=/home10T/bzx/workspace/KVCore uv run pytest tests/test_llm_engine.py tests/test_scheduler.py tests/test_model_runner_kv_boundary.py tests/test_triton_paged_attention.py -q
10 passed in 78.98s

PYTHONPATH=/home10T/bzx/workspace/KVCore uv run pytest -q
60 passed in 184.48s
```

## 1. 当前阶段判断

### 已确认进展

当前项目已经从“静态组件骨架”推进到“最小 engine/scheduler/model-runner 闭环”：

- `EngineCore.step()` 已经串起 `Scheduler.schedule()`、`ModelRunner.execute_model()`、`Scheduler.update_from_outputs()`。
- `Scheduler` 已经能处理 waiting prefill、running prefill continuation、decode priority，以及 `num_computed_tokens` 推进。
- `ModelRunner` 已经能初始化共享 KV tensor、构建 `PagedAttentionMetadata`、通过 forward context 执行模型 forward、取 sample position 并采样。
- CUDA 可用时默认选择 `triton_paged`，CPU 或非 CUDA 路径使用 `torch_paged` reference。
- `TritonPagedAttentionBackend` 已经实现 paged write 和基于 block table 的 causal paged attention，并有 prefill/decode cache 读取单测。

### 当前定位

KVCore 现在适合进入“真实模型最小 E2E 修通 + 语义对齐验证”阶段，还不适合做性能 claim 或系统论文式对比。

原因是：当前单测确认了组件可运行，但还没有证明真实 Llama-3.1-8B 在多层、多头、真实 tokenizer、真实权重、真实 prompt 下稳定生成，也没有建立与 HF/vLLM 输出一致性的回归基准。

## 2. 主要问题

### 问题 A：live 8B 测试的 KV block 配置可能不足

确认事实：

- `BlockPool` 会从 `num_gpu_blocks` 里取出 1 个 `null_block`，所以可用物理块数是 `num_gpu_blocks - 1`。
- `KVManager` 为每层创建一个 `SingleTypeKVManager`，但它们共享同一个全局 `BlockPool`。
- `tests/test_llm_engine_live.py` 当前设置 `block_size=16`、`num_gpu_blocks=256`、`max_num_scheduled_tokens=128`、`max_model_len=512`。
- Llama-3.1-8B 通常是 32 层。若一次 prefill chunk 为 128 token，则每层需要 `ceil(128 / 16) = 8` 个 block，32 层共需要 256 个新 block，但全局池只有 255 个可用 block。

影响：

- live 测试可能在首个 128-token prefill chunk 就无法调度，`LLMEngine.generate()` 可能报 “Engine made no scheduling progress”。
- 即使当前 prompt 较短没有触发，也说明 `max_model_len=512` 与 `num_gpu_blocks=256` 的语义不一致。32 层、512 token 至少需要 `32 * ceil(512 / 16) + 1 = 1025` 个全局 block。

结论：

- 这是配置层面的直接风险，不是 Triton kernel 本身的问题。
- live smoke test 可以先降 `max_num_scheduled_tokens` 或增大 `num_gpu_blocks`。若要支持 512 token 的 full attention，应把 `num_gpu_blocks` 调到至少 1025，最好留出余量。

### 问题 B：TorchSDPA 路径不是 paged KV 语义（已删除）

确认事实：

- 原始 `TorchSDPAAttentionBackend` 已删除。
- CPU tiny engine 现在使用 `torch_paged`，因此也会走 `PagedAttentionMetadata.kv_cache_tensor`、`block_tables`、`slot_mapping` 语义。

影响：

- CPU fallback 不再绕过 paged KV 语义。
- 代价是 `torch_paged` reference 很慢，只适合小模型/小 batch correctness。

结论：

- 已增加低性能但语义正确的 PyTorch paged backend，作为 Triton backend 的参考实现。
- 它不追求速度，只负责按 `slot_mapping` 写 KV、按 `block_tables` gather 历史 K/V，并与普通 dense attention 对齐。

### 问题 C：Triton paged backend 有基础单测，但覆盖面仍偏窄

确认事实：

- 当前 Triton 单测覆盖了单请求、单 KV head、head_dim=16、block_size=2 的 prefill 与 decode cache reuse。
- `TritonPagedAttentionBackend.forward()` 要求 flattened runtime input 的 batch size 为 1，内部把 token 维度作为 `num_tokens`。
- kernel 使用 `token_request_indices` 和每层 block table 支持多请求元数据，但现有测试没有覆盖多请求 batch、多 KV head、GQA、多层真实模型输出对齐。

风险：

- 多请求连续 batching、不同 request position、GQA head 映射、较大 head_dim、跨 block 边界等场景可能隐藏 bug。
- 当前没有与 dense reference 做整模型级别 logits 对齐。

结论：

- Triton backend 已经不是空壳，但还处在 “kernel smoke + 局部语义” 阶段。
- 下一步必须补齐 reference backend 和更系统的 kernel correctness matrix。

### 问题 D：模型接口仍保留旧 `kv_caches` 兼容层（已移除）

确认事实：

- Llama/Qwen/Mistral 模型 forward 已不再接收 `kv_caches` 参数。
- 新主路径通过 `ForwardContext(PagedAttentionMetadata)` 注入共享 KV tensor 和 paged metadata，模型顶层 forward 不再显式接收 `attn_metadata`。
- 旧 `RecordingKVCache.update()` 风格测试已迁移为 fake attention backend 或 paged metadata 测试。

影响：

- 框架边界已收敛到单一 shared KV tensor 路径。
- 新功能测试需要继续强制使用 `PagedAttentionMetadata`。

结论：

- 不再保留 `kv_caches` 兼容参数。
- 后续如需 debug backend，应通过 `AttentionBackend` 注入 fake/reference backend，而不是恢复 per-layer KV cache 参数。

### 问题 E：文档状态需要更新（已更新）

确认事实：

- 旧版 `notes/current-architecture-flow.md` 曾写着 “No real scheduler-to-runner execution loop yet”。
- 实际代码已经有 `EngineCore.step()` 和 `ModelRunner.execute_model()`。

影响：

- 后续开发者会误判当前进度，把已经完成的 step-level API 当成缺口。

结论：

- 已更新架构文档，把当前状态改成：已有最小执行循环，但真实模型 paged correctness 和性能验证仍未完成。

## 3. 下一步开发路线

### P0：先修 live 8B smoke test

问题陈述：当前最需要证明真实模型至少能完成 1 token generation。

假设：主要阻塞来自 KV block 配置不足、导入/加载耗时、或真实 Triton path 的形状/数值 bug。

方法：

1. 把 live 测试拆成直连脚本，分别计时：
   - import
   - tokenizer load
   - model load
   - KV init
   - first prefill/decode step
2. 调整配置：
   - smoke：`max_num_scheduled_tokens=32`，`num_gpu_blocks>=257`
   - 512-token full attention：`num_gpu_blocks>=1025`
3. 在失败时打印：
   - prompt token 数
   - num layers
   - block size
   - required blocks per step
   - free blocks
   - scheduler output summary

预期收益：

- 快速区分环境 I/O 问题、容量问题、kernel 问题。
- 形成稳定的 “真实模型最小闭环”。

风险：

- 8B 权重加载受文件系统影响较大，pytest 形式可能仍不稳定。

最小验证计划：

```text
python scripts/run_live_llama31_smoke.py --max-tokens 1 --max-scheduled-tokens 32 --num-gpu-blocks 384
```

通过标准：

- 能返回 1 个 token。
- 记录每阶段耗时。
- 不出现 no progress、CUDA shape error、Triton compilation error。

### P1：实现 PyTorch paged reference backend（已完成最小版）

问题陈述：当前缺少可读、可调试、语义正确的 paged attention reference。

假设：有 reference backend 后，Triton kernel 和整模型 logits 对齐会更容易定位。

方法：

1. 新增 `TorchPagedAttentionBackend`：
   - 输入同 `TritonPagedAttentionBackend`
   - 按 `slot_mapping[layer_idx]` 写入 `kv_cache_tensor`
   - 按 `block_tables[layer_idx]` gather `0..position` 的历史 K/V
   - 用 PyTorch softmax/matmul 计算 causal attention
2. 将测试分三层：
   - backend-level：torch paged vs dense reference
   - kernel-level：triton paged vs torch paged
   - model-level：tiny model 的 torch paged vs triton paged logits 对齐

预期收益：

- 后续所有 KV 优化都有稳定 oracle。
- 避免把非 paged dense attention 误当 paged cache reference。

风险：

- reference backend 很慢，只能用于小 batch、小 seq、小模型测试。

最小验证计划：

```text
pytest tests/test_torch_paged_attention.py tests/test_triton_paged_attention.py -q
```

通过标准：

- 单请求 prefill/decode 对齐 dense attention。
- 多请求 batch 对齐 dense attention。
- GQA 场景对齐 dense attention。

当前实现状态：

- 已新增 `kvcore/model/attn_backend/torch_paged.py`。
- 已注册 backend 名称 `torch_paged`。
- 已增加 `tests/test_torch_paged_attention.py`，覆盖多请求 + GQA dense reference 和 decode cache reuse。

### P2：扩大 Triton paged correctness matrix（已完成第一组对齐）

问题陈述：当前 Triton 单测只覆盖很窄的形状组合。

假设：多数潜在 bug 会出现在多请求、GQA、跨 block 和不同 head_dim 上。

方法：

- 覆盖参数：
  - `num_reqs`: 1, 2, 4
  - `query_len`: prefill 1/3/17，decode 1
  - `block_size`: 2, 16
  - `num_query_heads / num_kv_heads`: 1, 4
  - `head_dim`: 16, 64, 128
- 与 `TorchPagedAttentionBackend` 对齐。

预期收益：

- 在做性能优化前先锁住语义。

风险：

- Triton 编译矩阵太大时 CI 变慢。可以把大矩阵标记为 CUDA/nightly。

最小验证计划：

```text
pytest tests/test_triton_paged_attention.py -q
```

通过标准：

- smoke matrix 默认跑少量关键组合。
- extended matrix 用 marker 单独跑。

当前实现状态：

- `tests/test_triton_paged_attention.py` 已增加 Triton vs Torch paged 的多请求 + GQA 对齐测试。
- 仍建议后续继续扩展 head_dim=64/128、block_size=16、prefill continuation、混合 prefill/decode batch。

### P3：整理 engine/scheduler 的容量与 admission control（已完成 profile 基础）

问题陈述：现在 `num_gpu_blocks` 是全局物理块数，但用户很容易按“单层 token capacity”理解。

假设：显式容量估算可以减少 live test 和实验配置错误。

方法：

1. 增加 helper：
   - `estimate_required_blocks(num_layers, block_size, seq_len, num_seqs)`
   - `estimate_max_tokens(num_layers, block_size, num_gpu_blocks, num_seqs)`
2. 在 `EngineCore` 初始化或 add_request 时给出明确错误：
   - 当前配置最多支持多少 token
   - 当前请求/step 需要多少 block
3. `Scheduler` 的 no-progress 错误补充 free block 和 required block 信息。

预期收益：

- 配置错误变成可解释错误。
- 后续实验更容易复现。

风险：

- 对 sliding window、prefix cache、hybrid KV 的估算后续会更复杂；第一版只声明 full attention 假设。

最小验证计划：

```text
pytest tests/test_scheduler.py tests/test_llm_engine.py -q
```

通过标准：

- block 不足时错误信息可解释。
- 现有 60 个测试不回退。

当前实现状态：

- `KVCoreConfig.runtime` 已支持 `profile_kv_cache=True` 或 `num_gpu_blocks=None`；旧 `EngineConfig` 作为兼容入口会转换到 `KVCoreConfig`。
- `ModelRunner.profile_run()` 会生成 `KVCacheProfileResult`，记录块数、单块字节数、显存预算和当前配置可支撑的单序列 token 上限。
- CPU 路径使用 `max_model_len` 和层数估算最小 block 数；CUDA 路径使用 `torch.cuda.mem_get_info()` 和 `gpu_memory_utilization` 估算。

### P4：更新文档和实验基线

问题陈述：文档和代码状态已有偏差，且缺少“正确性基线”说明。

方法：

1. 更新 `notes/current-architecture-flow.md`：
   - 改为已有最小 scheduler-runner loop。
   - 明确原始 `torch_sdpa` backend 已删除。
   - 明确 `triton_paged` 是当前 CUDA paged runtime path。
2. 增加 `notes/live-e2e-validation-plan.md`：
   - 环境
   - 模型
   - prompt
   - 配置
   - 通过/失败日志模板

预期收益：

- 后续开发和论文实验更容易追踪阶段性结论。

风险：

- 文档会随着代码快速过期，需要在关键 PR 后同步更新。

## 4. 适合暂缓的方向

以下方向很重要，但不建议现在优先做：

- 性能 benchmark：真实正确性还未充分锁定，先报 latency/throughput 容易误导。
- CPU offload / hierarchical storage：需要稳定的 paged KV runtime 和 block lifecycle 事件。
- speculative decoding：会放大 scheduler、KV rollback、sampling 状态复杂度。
- sliding window cache hit：当前 full attention E2E 还没有完全稳定。
- prefix cache 性能优化：先保证 prefix cache 语义、block hash、cache hit 的 end-to-end correctness。

## 5. 推荐的一周内任务拆分

1. Day 1：修 live smoke 配置和直连脚本，记录真实失败点。
2. Day 2：实现 `TorchPagedAttentionBackend` reference。
3. Day 3：补 Triton vs Torch paged 对齐测试，覆盖 GQA 和多请求。
4. Day 4：跑通 Llama-3.1-8B 1-token smoke，并保存日志。
5. Day 5：更新架构文档，建立 correctness checklist。

## 6. 结论

当前 KVCore 的架构方向是合理的：`Scheduler` 管逻辑 KV 生命周期，`ModelRunner` 管物理 KV tensor 和 runtime metadata，attention backend 消费 paged metadata。这和 vLLM 的核心分层一致，同时比 vLLM 更适合研究型快速迭代。

下一步最关键的不是继续扩大功能面，而是把 “真实模型 + paged KV + decode reuse” 的正确性闭环钉牢。建议优先修 live 8B smoke、建立 PyTorch paged reference backend，再扩展 Triton correctness matrix。完成这三步后，才适合进入吞吐/延迟/显存效率评估和更研究化的 KV 策略实验。
