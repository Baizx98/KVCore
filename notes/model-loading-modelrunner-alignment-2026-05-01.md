# Model Loading and ModelRunner Alignment - 2026-05-01

## Problem Statement

KVCore 的 8B 模型初始化阶段主要瓶颈来自权重加载和 runner-side 元数据准备。原实现的问题有两类：

- 权重加载：多 shard safetensors/PT 文件按文件串行读取，且旧顺序会先把随机初始化模型搬到 GPU，再加载真实权重，GPU 峰值更高。
- ModelRunner 元数据：`ModelStepOutput.req_id_to_index`、`ModelRunnerPreparedInput.num_reqs`、`num_scheduled_tokens`、`max_query_len`、`max_seq_len` 等字段可以由已有 request/tensor 状态推导，不需要作为独立数据长期保存。

## External References

- vLLM latest default loader keeps the same high-level path: prepare weight files, pick safetensors/PT iterator, then call model-specific `load_weights`. Its loader supports multithread safetensors/PT shard iteration through extra config.
  - https://docs.vllm.ai/en/stable/api/vllm/model_executor/model_loader/default_loader/
- vLLM latest `multi_thread_safetensors_weights_iterator` loads safetensors shards on CPU with a thread pool and pops tensors from the shard state dict to reduce retained memory.
  - https://docs.vllm.ai/en/latest/api/vllm/model_executor/model_loader/weight_utils/
- Transformers `load_state_dict` defaults checkpoint loading to CPU, uses safetensors mmap when possible, and only moves tensors to `map_location` when requested.
  - https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
- vLLM V1 GPU model runner keeps persistent `InputBatch`, updates request rows from `SchedulerOutput`, commits block table early, and derives per-step token positions from persistent batch state plus scheduled-token counts.
  - https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_model_runner/

## Method

### Weight Loading

Implemented in `kvcore/model/model_loader/default_loader.py`:

- Added default multithread loading for multi-shard safetensors/PT checkpoints.
- Default thread count is `4`, controlled by `KVCORE_WEIGHT_LOAD_THREADS`.
- Can disable with `KVCORE_DISABLE_MULTITHREAD_WEIGHT_LOAD=1`.
- Safetensors multithread path loads shard state dicts on CPU, matching vLLM/Transformers behavior and avoiding concurrent GPU shard allocation.
- Bounded the number of in-flight shard futures to `max_workers`, so the loader does not eagerly retain every shard state dict.
- Changed `load_model()` order to load weights before moving the model to the target device. This avoids putting randomly initialized weights on GPU before real weights are loaded.
- The single-file path is preserved and still chooses CPU/GPU tensor reads based on the model's current device.

### ModelRunner Metadata

Implemented in `kvcore/model/model_runner.py`:

- Removed `ModelStepOutput.req_id_to_index`; engine only needs `(req_ids, sampled_token_ids)` to build the sampled-token map.
- Converted derived fields in `ModelRunnerPreparedInput` into properties:
  - `num_reqs`
  - `num_scheduled_tokens`
  - `max_query_len`
  - `max_seq_len`
- Kept `num_prefill_reqs` and `num_decode_reqs` because current attention metadata still needs them and they are not derivable from `context_lens` alone during chunked prefill.
- Kept persistent `InputBatch` as the owner of request rows, token ids, block ids, computed-token counts, and sampled output history.
- Updated `scripts/run_offline_batch_debug.py` to print the current `scheduled_new_reqs` / `scheduled_cached_reqs` scheduler contract instead of removed scheduled-request compatibility fields.

## Validation

### Unit / Targeted Tests

Command:

```bash
uv run pytest tests/test_model_loading.py tests/test_model_runner_kv_boundary.py tests/test_scheduler.py tests/test_request_queue.py
```

Result:

```text
26 passed in 149.50s
```

After the CPU-first load-order fix, `tests/test_model_loading.py` was rerun:

```text
7 passed in 3.06s
```

### End-to-End Test

Command:

```bash
KVCORE_WEIGHT_LOAD_THREADS=4 uv run python scripts/run_llm_engine_offline_batch.py \
  --model /Tan/model/Llama-3.1-8B-Instruct \
  --device cuda:0 \
  --attn-backend triton_paged \
  --max-num-seqs 4 \
  --max-num-scheduled-tokens 128 \
  --log-level INFO
```

Log:

```text
logs/llm_engine_offline_batch_llama31_8b_2026-05-01_15-11-31.log
```

Result:

```text
Loading safetensors shards with multithread CPU loader files=4 threads=4
Model weights loaded params=195 elapsed=3.697s
engine_init_sec=59.088
generate_sec=13.042
total_elapsed_sec=72.132
outputs=7
```

The first output finished by stop token:

```text
offline-000: 中国的首都是北京。
```

The other six requests finished by length, which matches their `max_tokens` settings in the offline batch script.

## Limits

- This is an initialization optimization, not a serving-throughput optimization.
- End-to-end timing is hardware and cache-state dependent. The successful run used PyTorch-visible `cuda:0` = NVIDIA L40S.
- A previous attempt on PyTorch-visible `cuda:1` = NVIDIA RTX A6000 loaded successfully but failed in the Triton backend with `device kernel image is invalid`; this appears to be backend/device compilation compatibility, not model loading.
- A previous attempt on PyTorch-visible `cuda:2` = NVIDIA GeForce RTX 3090 failed before the fix because moving the randomly initialized 8B model to a 24GB GPU exhausted memory.

## Conclusion

The loader is now closer to vLLM/Transformers practice for sharded checkpoint initialization: CPU-side shard loading, bounded multithread reading, and one final model move to target device. ModelRunner metadata is also slimmer: per-step scalar fields that are mechanically derivable from tensors/request order are properties rather than stored dataclass fields, while attention-critical prefill/decode counts remain explicit.
