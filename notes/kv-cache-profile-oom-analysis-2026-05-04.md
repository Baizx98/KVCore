# KV Cache Profile OOM Analysis - 2026-05-04

## Problem

`scripts/run_llm_engine_offline_batch.py` failed during formal KV cache tensor allocation after the new dummy-forward profile path:

- Model: `/Tan/model/Llama-3.1-8B-Instruct`
- Device: `cuda:0`
- `max_num_seqs=1024`
- `max_num_scheduled_tokens=1024`
- Profile result: `num_gpu_blocks=405847`
- Allocation failure: tried to allocate `24.77 GiB`

## Evidence

Profile log:

```text
free=8230797312
total=47576711168
peak=16225650688
non_kv_peak=16221390848
budget=42819040051
available_for_kv=26597649203
bytes_per_block=65536
blocks=405847
```

Allocation size:

```text
405847 blocks * 65536 bytes/block = 26,597,588,992 bytes = 24.77 GiB
```

But the CUDA allocator reported only about `7.57 GiB` free at the actual allocation point.

## Root Cause

The profile formula currently uses:

```text
memory_budget = total_gpu_memory * gpu_memory_utilization
available_for_kv = memory_budget - non_kv_peak_memory
```

This assumes KVCore can use 90% of the full GPU capacity. That is invalid in this run because the GPU was already heavily occupied after model load and/or by other processes. The profile log itself says only `8.23 GB` was free before dummy forward.

So the estimator produced a theoretical KV budget of about `26.60 GB`, but the allocator could only satisfy about `7.6-8.2 GB` at runtime. The resulting `405847` blocks therefore oversubscribed current free memory by roughly `18-19 GiB`.

## Secondary Observation

`peak=16.23 GB` is close to the model/non-KV PyTorch footprint during the dummy forward. The temporary profile KV cache is tiny here:

```text
profile_tokens=1024
block_size=16
profile_kv_blocks=65
65 * 65536 bytes = 4.06 MiB
```

So `non_kv_peak` is essentially the model + dummy forward activation/workspace footprint. The profile did capture activation overhead, but the final capacity calculation used the wrong upper bound for an already occupied GPU.

## Fix Direction

For this repo's current single-process offline runner, the safer formula should be bounded by current free memory:

```text
available_for_kv = free_memory_at_profile_start * gpu_memory_utilization
```

or, if keeping a vLLM-like total-memory budget, clamp it by current free memory:

```text
requested_budget = total_gpu_memory * gpu_memory_utilization - non_kv_peak_memory
free_budget = free_memory_at_profile_start * gpu_memory_utilization
available_for_kv = min(requested_budget, free_budget)
```

The second option preserves the dummy-forward activation accounting while preventing allocation beyond the memory that is actually free for this run.

With the logged free memory, the upper bound would be approximately:

```text
8,230,797,312 * 0.9 / 65,536 = about 113,000 blocks
```

That corresponds to about `6.9 GiB` KV tensor allocation, which fits the observed free memory much better than `24.77 GiB`, but it still leaves almost no room for post-KV runner buffers.

The next failure after this clamp happened in `InputBatch` initialization. With the logged config, the formal per-layer block tables alone need:

```text
32 layers * 1024 reqs * ceil(131072 / 16) blocks/req * 4 bytes = 1.0 GiB
```

So the KV tensor budget also needs to reserve post-KV static metadata memory.

## Conclusion

The OOM is not caused by dummy forward itself. It is caused by the post-profile capacity formula using total GPU memory instead of respecting current free memory. Dummy forward correctly adds activation/workspace awareness, but the final block count must be clamped by free memory for shared or already occupied GPUs.

## Implemented Change

`ModelRunner._estimate_num_gpu_blocks_with_profile()` now computes:

```text
total_memory_budget = total_memory * gpu_memory_utilization
post_kv_cache_memory = block tables + slot mappings + step buffers
free_memory_budget =
    free_memory_at_profile_start * gpu_memory_utilization
    - post_kv_cache_memory
    - 256 MiB safety margin
requested_budget = total_memory_budget - non_kv_peak_memory
available_for_kv = min(requested_budget, free_memory_budget)
```

This keeps the dummy-forward activation/workspace accounting while preventing the formal KV cache tensor allocation from exceeding the memory that is actually free in the current process environment.
