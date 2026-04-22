# Request 最小数据结构设计记录

日期: 2026-04-21 00:35

## problem statement

当前 `kvcore/utils/request.py` 只有空 `Request` 骨架，无法支撑后续 scheduler / KV manager / model runner 的请求生命周期管理。

目标是参考 vLLM 的 `Request`，先实现一个不依赖 vLLM 外部类型的最小版本。

当前框架只支持 decoder-only generation request：

- 不支持 `prompt_embeds`
- 不支持 pooling request
- sampling 参数独立放在 `kvcore/utils/sampling_params.py`

## hypothesis

如果先保留请求生命周期中最核心的状态：

- prompt token ids
- output token
- request status
- token 计数
- completion reason
- priority ordering
- event queue
- block hash hook

就可以支撑后续最小 scheduler 和 KV cache 管理逻辑，而不需要过早引入 LoRA / 多模态 / pooling 等完整复杂度。

## method

本次新增：

- `Request`
- `RequestStatus`
- `FinishReason`
- `SamplingParams`
- `RequestEvent`
- `StreamingUpdate`

保留的 vLLM 语义：

- `WAITING / RUNNING / PREEMPTED / FINISHED_*`
- `append_output_token_ids`
- `num_tokens`
- `num_output_tokens`
- `num_tokens_with_spec`
- `is_finished`
- `get_finished_reason`
- `record_event`
- `take_events`
- `take_prefill_stats`
- `__lt__` priority ordering
- `block_hasher` hook

主动删减的复杂项：

- LoRA
- structured output
- multi-modal encoder inputs
- prompt embeds
- pooling request
- PrefillStats 复杂对象
- EngineCoreEvent 复杂对象
- vLLM SamplingParams 直接依赖

## experiment

新增测试：

- `tests/test_request.py`

覆盖内容：

1. generation request 初始化
2. output token append 和计数更新
3. 空 prompt token ids 校验
4. sampling prefix cache flag
5. event queue 消费
6. finished status 和 finish reason
7. priority ordering
8. streaming update
9. block hasher hook

执行结果：

- `ruff check`: 通过
- `pytest tests/test_request.py -q`: `10 passed in 1.34s`
- `compileall`: 通过

## conclusion

当前 `Request` 已经具备后续实现最小 scheduler / KV lifecycle 所需的基础状态管理能力。

下一步可以围绕它继续补：

- scheduler queue
- block table allocation input
- prefill/decode 阶段状态
- request finish / abort path
