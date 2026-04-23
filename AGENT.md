# AGENT.md

本文件定义了代码智能体在仓库级别的工作规则。

这是策略文件，不是设计文档。
架构细节应写在 `docs/ARCHITECTURE.md`。
实现规划应写在 `docs/IMPLEMENTATION_STAGES.md`。

---

## 1. 仓库定位

请将本仓库视为一个**面向研究、以 KV 为中心的 LLM 推理框架**。

核心关注点：

- 以 KV cache 作为运行时核心抽象
- 按层执行（layer-by-layer execution）
- 便于研究迭代与重构

不要将本仓库当作生产级服务栈。

---

## 2. 当前工程约束

在当前阶段，项目应保持 **Python-first**。

规则：

- 优先采用纯 Python 实现
- 工具链保持轻量
- 避免过早引入复杂的原生构建系统
- 如确实需要自定义 GPU kernel，优先考虑 **Triton**
- 未经用户明确说明，不要引入 C++/CUDA 扩展基础设施
- 项目必须使用项目目录下的.venvpython环境
- 在note文件夹中添加新的md文件时，在开头添加日期和时间，语言使用中文

这意味着：

- 不要过早搭建 CMake
- 不要过早引入 `pybind11` / 自定义扩展构建流水线
- 不要做出“默认必须依赖原生扩展”的架构决策

---

## 3. 范围约束

当前支持的模型家族：

- Qwen3
- Llama3
- Mistral3

当前目标功能范围：

- PagedAttention
- Prefix Cache
- Continuous Batching
- Chunked Prefill
- CPU Offload
- CUDA Graph
- selective block attention
- block-granularity pruning
- hook-based KV lifecycle control

除非用户改变方向，以下内容明确不在范围内：

- beam search
- parallel sampling
- speculative decoding
- LoRA
- tensor / pipeline / expert parallelism
- distributed multi-node inference
- training or fine-tuning

---

## 4. 设计与实现规则

在编写或修改代码时：

- 严格遵循vllm架构的设计
- 在概念上保持 KV metadata 与 KV data 分离
- 默认使用 block/page 粒度作为管理单元
- 保持按层执行的控制流
- 在可行情况下，将调度策略与 KV 策略解耦
- 相比过早优化，更优先可读性、可测试性和可观测性
- 保持改动原子化、易于审查

若当前结构阻碍清晰实现，允许进行重构。

---

## 5. 文档策略

仓库内所有文档均为**活文档（living documents）**：

- `AGENT.md`
- `README.md`
- `docs/` 目录下文件

不要假设任何现有文档永久正确。

必须遵循：

- 持续对比代码、文档与用户意图
- 若实现使文档表述失效，需更新对应文档
- 不要强行让代码迁就过期文档
- 当仓库现实发生变化时，`AGENT.md` 本身也可修订

当出现冲突时，优先级如下：

1. 用户最新的明确指令
2. 更清晰、可维护的实现方向
3. 现有仓库文档

---

## 6. 开发流程期望

除非用户另有要求，使用仓库既有工具链：

- `uv`
- `ruff`
- `mypy`
- `pytest`
- `pre-commit`

新增检查或工具时：

- 从低摩擦方式开始
- 优先渐进式收紧，而非默认严格
- 在代码库稳定前避免引入重量级基础设施

提交信息应遵循 `docs/DEVELOPMENT.md` 中记录的仓库规范。
优先保证每次提交都是一个有意义的原子改动。
