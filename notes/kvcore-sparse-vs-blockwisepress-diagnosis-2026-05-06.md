# KVCore sparse KV 与 kvpress-study BlockWisePress 对齐诊断

日期：2026-05-06

## 结论

当前 KVCore 稀疏效果差，主要不是单个参数调得不好，而是当前实现还不是 `kvpress-study` 中 `BlockWisePress` 的等价系统实现。

当前 KVCore 实现更接近：

- prefill 默认不压缩；
- decode 阶段周期性生成 sparse read block table；
- score 来自当前 decode query window 与 resident KV block summary 的近似相关性；
- KV 物理块默认仍保留，主要做 compute sparsity；
- E2E 为了稳妥使用了低压缩率和较长 interval。

而 `BlockWisePress` 主路径是：

- 在 prefill 期间直接对完整 prompt KV 做 block-level 压缩；
- 用 prompt tail query window 对所有 block 一次性打分；
- 直接 gather kept token indices，生成压缩后的 KV；
- 后续 decode 在压缩 KV 上运行。

因此二者目前只能说“共享一部分 block summary 参数”，不能说配置和行为保持一致。

## 当前 KVCore 配置与行为

`SparseKVConfig` 默认值：

- `mode=disabled`
- `selection_interval=step`
- `compression_ratio=0.5`
- `q_window_size=32`
- `prefix_sink_blocks=1`
- `protected_recent_blocks=2`
- `score_ema_alpha=0.8`
- `summary_topk_keys=4`
- `mean_key_weight=0.75`
- `enable_prefill_sparsity=False`
- `enable_decode_sparsity=True`

证据：`kvcore/config.py:96-157`。

离线 E2E 脚本默认将 sparse ratio 改成了 `0.2`，prefill sparse 仍默认关闭，decode sparse 默认开启。

证据：`scripts/run_llm_engine_offline_batch.py:148-168`、`scripts/run_llm_engine_offline_batch.py:256-269`。

实际选择逻辑：

- 没有分数时直接 dense：`_select_selected_block_indices()` 在 `scored_candidates` 为空时返回全部 valid blocks。
- keep budget 按 `ceil(valid_blocks * (1 - compression_ratio))` 计算。
- sink、recent、当前写入 block 会被保护。
- `selection_interval=n_tokens` 使用全局 `step_id % interval_tokens == 0`，不是按 request 的 decode token 数独立触发。

证据：`kvcore/kv/kv_manager.py:508-582`、`kvcore/kv/kv_manager.py:584-604`。

score 逻辑：

- 记录 Attention wrapper 中已经投影并完成 RoPE 前后路径中的 query tensor；
- 将 query heads 平均到 KV heads；
- 每个 block 用 `mean_keys` 与 `norm topk key mean` 做 summary；
- score 对 query window 和 head 直接取 mean；
- 没有 `query_agg_mode=max/topr_mean`，没有 `head_agg_mode=strength_weighted/top_head_only`，没有 `multi_rep_max/adaptive_fusion_v1`。

证据：`kvcore/model/layer/attention.py:80-83`、`kvcore/model/block_score.py:44-85`、`kvcore/model/block_score.py:107-151`。

## kvpress-study BlockWisePress 主路径

`BlockWisePress` 默认核心参数：

- `compression_ratio=0.0`
- `block_size=16`
- `q_window_size=32`
- `summary_topk_keys=4`
- `mean_key_weight=0.75`
- `prefix_sink_blocks=1`
- `protected_recent_blocks=2`
- `summary_mode=mean_plus_norm_topk_mean`
- `representative_mode=key_norm`
- `query_agg_mode=mean`
- `head_agg_mode=uniform_mean`
- `representative_k=4`
- `multi_rep_k=4`

证据：`/home10T/bzx/workspace/kvpress-study/kvpress/presses/block_wise_press.py:31-61`。

实际实验脚本中，stage3 主线已经将 `query_agg_mode` 设为 `max`，`q_window_size=64`，`query_topr=16`，并比较 `blockwise_main`、`blockwise_multi_rep`、`adaptive_fusion_v1` 等变体。

证据：`/home10T/bzx/workspace/kvpress-study/evaluation/run_blockwise_stage3_ratio70_fraction20_primarybench.py:266-286`、`/home10T/bzx/workspace/kvpress-study/evaluation/run_blockwise_stage3_ratio70_fraction20_primarybench.py:332-337`。

已有分析笔记也明确记录：`query_agg=max` 应作为 blockwise 主线默认项，`blockwise_main` 是稳健通用配置，`blockwise_multi_rep` 是更强的检索/多峰候选。

证据：`/home10T/bzx/workspace/kvpress-study/note/blockwise_stage3_current_results_analysis_and_next_steps_zh.md:193-200`。

## 关键不一致点

### 1. prefill 压缩 vs decode 计算稀疏

`BlockWisePress.compress()` 在 prefill 中直接 build plan，并 gather kept token indices，返回压缩后的 keys/values。

证据：`/home10T/bzx/workspace/kvpress-study/kvpress/presses/block_wise_press.py:531-563`。

KVCore 当前 E2E 主要跑的是 decode dynamic sparse。prefill 产生的完整 KV 仍保留，后续 decode 只是让部分 block 在当前步不可见。

这会导致两个差异：

1. 如果 prompt 很短或 decode 很短，能省的 attention 计算本来就少。
2. 如果 sparse plan 到 decode 后期才触发，绝大部分 prefill 成本已经发生。

### 2. query 来源不一致

`BlockWisePress` 使用 `get_prerope_query_states(module, hidden_states[:, -q_window:])`，即从 hidden states 重新经过 q projection，得到 pre-RoPE query。

证据：`/home10T/bzx/workspace/kvpress-study/kvpress/presses/block_wise_press.py:343-345`、`/home10T/bzx/workspace/kvpress-study/kvpress/utils.py:12-53`。

KVCore 当前从 Attention wrapper 中 record 的是进入 attention backend 前的 query tensor。这个 query 已经经过模型 attention 层内的 reshape 和位置编码路径影响。它可能更接近真实 attention query，但和 BlockWisePress 的 block score 并不严格一致。

### 3. query aggregation 不一致

KVCore 固定对 query window 做 mean：

```text
scores.mean(dim=(0, 1))
```

证据：`kvcore/model/block_score.py:144-151`。

BlockWisePress 支持 `mean/max/topr_mean/adaptive_mean_max_v1`，stage3 结论更偏向 `query_agg=max`。

证据：`/home10T/bzx/workspace/kvpress-study/kvpress/presses/blockwise_components.py:27-48`、`/home10T/bzx/workspace/kvpress-study/note/blockwise_stage3_current_results_analysis_and_next_steps_zh.md:193-200`。

这很可能是精度差的主要原因之一。mean 会稀释检索型任务中的少数强相关 query；max 更适合保留被少数 query 强命中的 block。

### 4. head aggregation 不完整

KVCore 当前把 GQA query heads 平均到 KV heads，然后对 head 维继续 mean。

证据：`kvcore/model/block_score.py:44-45`、`kvcore/model/block_score.py:153-164`。

BlockWisePress 的默认也是 `uniform_mean`，但它保留了 `strength_weighted` 和 `top_head_only`，并在实验中作为可控变量。

证据：`/home10T/bzx/workspace/kvpress-study/kvpress/presses/blockwise_components.py:51-71`。

如果当前任务依赖少数 head 的检索信号，KVCore 的平均会进一步稀释 block score。

### 5. summary mode 不完整

KVCore 只有 `mean_plus_norm_topk_mean` 的简化版本。

BlockWisePress 还支持：

- `mean_only`
- `norm_topk_mean_only`
- `multi_rep_max`
- `adaptive_fusion_v1`

证据：`/home10T/bzx/workspace/kvpress-study/kvpress/presses/blockwise_components.py:8-14`、`/home10T/bzx/workspace/kvpress-study/kvpress/presses/block_wise_press.py:253-319`。

已有 kvpress-study 结果显示 `blockwise_multi_rep` 在检索/多峰块候选上更强。KVCore 目前没有对应能力。

### 6. 实际 E2E 配置过于保守

上一轮 E2E 用的是：

- `compression_ratio=0.2`
- `selection_interval_tokens=16`
- `protected_recent_blocks=4`
- `prefix_sink_blocks=1`
- `q_window_size=64`

日志中实际在后期只从 `224` blocks 跳过 `32` blocks，等价压缩约 `14.3%`，不是配置名义上的 `20%`。这是因为 sink/recent/current block 保护和 ceil keep budget 会抬高实际保留量。

因此如果用户感知“稀疏效果差”指性能收益差，这是符合预期的：压缩触发晚、压缩率低、保护块多、且当前 Triton sparse read table 仍是逐 block 遍历，并没有做专门的稀疏吞吐优化。

### 7. step_id 触发语义可能不适合 batch

KVCore 的 `n_tokens` interval 基于 KVManager 全局 `step_id`，而不是 request-local decode token count。

证据：`kvcore/kv/kv_manager.py:526-528`。

在 batch 中，不同 request 进入 decode 的时间不同，使用全局 step 可能导致某些请求第一次拿到 score 后很久才被 sparse plan 消费，也可能导致短请求几乎没有有效稀疏阶段。

## 是否与 BlockWisePress 保持一致

不一致。

一致的部分：

- block size 默认 `16`；
- `q_window_size` 默认可设到 `32/64`；
- `summary_topk_keys=4`；
- `mean_key_weight=0.75`；
- `prefix_sink_blocks=1`；
- `protected_recent_blocks=2`；
- summary 主干类似 `mean_plus_norm_topk_mean`；
- compression ratio 的预算公式同为 `ceil(num_blocks * (1 - ratio))` 这一类块级 keep budget。

不一致的部分：

- BlockWisePress 是 prefill permanent compression；KVCore 当前主要是 decode compute sparsity。
- BlockWisePress 对完整 prompt 一次性打分；KVCore 用 step-wise query feedback 和 EMA。
- BlockWisePress stage3 主线偏向 `query_agg=max`；KVCore 固定 `mean`。
- BlockWisePress 支持多 summary mode；KVCore 只有简化 summary。
- BlockWisePress 支持 per-batch block score tensor；KVCore 当前 per request/layer/logical block 反馈。
- BlockWisePress 直接压缩 token-level KV；KVCore 保留完整 physical KV，只构造 sparse read block table。

## 建议的下一步

### 短期修正

1. 在 `SparseKVConfig` 中显式加入：
   - `summary_mode`
   - `representative_mode`
   - `query_agg_mode`
   - `head_agg_mode`
   - `representative_k`
   - `multi_rep_k`
   - `query_topr`
   - `head_topk`

2. 先把默认 experimental 配置改成 kvpress-study stage3 主线：
   - `block_size=16`
   - `q_window_size=64`
   - `summary_topk_keys=4`
   - `mean_key_weight=0.75`
   - `representative_k=4`
   - `multi_rep_k=4`
   - `query_topr=16`
   - `summary_mode=mean_plus_norm_topk_mean`
   - `representative_mode=key_norm`
   - `query_agg_mode=max`
   - `head_agg_mode=uniform_mean`

3. 把 `BlockScoreCollector._score_summary()` 从固定 mean 改成可配置聚合，至少支持 `mean/max/topr_mean`。

4. `selection_interval=n_tokens` 改成 request-local decode token interval，而不是全局 step interval。

### 中期验证

1. 做一个离线 oracle：同一段 prompt、同一层、同一 KV tensor，比较 KVCore 选出的 block indices 与 kvpress-study `BlockWisePress.build_block_plan()` 的 Jaccard。
2. 先只对齐 `blockwise_main`，不要一上来实现所有 stage3 变体。
3. E2E 不要只看输出文本，要记录：
   - per request/layer selected blocks；
   - actual compression ratio；
   - sparse plan 首次触发 step；
   - dense vs sparse decode latency；
   - block selection Jaccard against BlockWisePress oracle。

## 当前判断

如果“稀疏效果差”指输出质量差，最可疑的是 `query_agg=mean` 与缺少 `multi_rep/max` 类 summary，导致检索型关键 block 被低估。

如果“稀疏效果差”指性能收益差，最可疑的是当前 E2E 实际跳过块太少、触发太晚，并且 sparse Triton kernel 只是支持 compact read table，还没有专门优化稀疏访存和 block scheduling。

下一步应该先做 BlockWisePress oracle 对齐，而不是继续盲调 compression ratio。
