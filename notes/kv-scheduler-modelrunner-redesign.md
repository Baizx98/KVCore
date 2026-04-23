# KV Scheduler And ModelRunner Redesign

Date: 2026-04-23 20:55:28 CST

## Problem Statement
KVCore removes vLLM's executor and worker layers, so we need a clear replacement for their ownership boundaries. The previous implementation put both logical `KVManager` initialization and physical KV cache tensor creation in `ModelRunner.initialize_kv_cache()`. That is too worker-like and conflicts with the desired architecture: Scheduler owns logical KV block allocation, ModelRunner owns model execution.

## vLLM Reference Boundary
In vLLM V1, scheduling code owns logical KV allocation. `KVCacheManager` wraps block lifecycle and exposes `get_computed_blocks`, `allocate_slots`, `free`, `get_blocks`, and `get_block_ids`; it constructs a coordinator and shared block pool internally. See `vllm/v1/core/kv_cache_manager.py` in upstream: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/kv_cache_manager.py

The single-type KV manager is a lower-level per-cache-type block lifecycle component. It stores `req_to_blocks`, `num_cached_block`, `new_block_ids`, and operates on the shared `BlockPool`, which matches our simplified per-layer manager direction. See upstream: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/single_type_kv_cache_manager.py

The scheduler path is where requests are admitted and KV slots are allocated before execution. Upstream scheduler lives under `vllm/v1/core/sched/scheduler.py`: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/sched/scheduler.py

The GPU model runner is responsible for model execution and worker-side runtime inputs rather than scheduler-owned request admission. Upstream GPU model runner is here: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu_model_runner.py

## KVCore Boundary
Because KVCore has no executor and no worker, responsibilities collapse into two main runtime actors:

- `Scheduler`: owns request queues, scheduling decisions, and logical KV block lifecycle through `KVManager`.
- `ModelRunner`: owns model creation/loading, physical KV cache tensors, attention-layer binding, block table construction, slot mapping, attention metadata construction, forward, logits, and sampling integration.

## Initialization Order
The corrected order should be:

```text
ModelRunner created
  -> create/load model
  -> Scheduler(kv_manager_config) creates KVManager
  -> ModelRunner.initialize_kv_cache(kv_manager_config)
  -> scheduler admits requests and calls KVManager.allocate_slots(...)
  -> ModelRunner.build_attention_metadata(kv_manager, scheduled_request_ids, sparse_plan)
  -> ModelRunner executes model forward
  -> Scheduler updates request/KV state after output
```

Important detail: `kv_manager_config` is shared configuration, but not shared ownership. Scheduler uses it to build `KVManager`; ModelRunner uses it to allocate one shared physical KV tensor shaped like:

```text
[2, num_gpu_blocks, block_size, num_kv_heads, head_size]
```

`num_gpu_blocks` is the global physical block count across all layers. Attention
metadata carries per-layer block ids, and those ids index directly into the
global tensor.

## Interface Changes
`Scheduler` now directly owns:

- `kv_manager: KVManager`

`ModelRunner.initialize_kv_cache(kv_manager_config)` now only initializes one shared physical KV tensor. It does not bind per-layer cache tensors to attention layers, and it does not create or retain a `KVManager`.

`ModelRunner.build_block_tables`, `build_slot_mapping`, and `build_attention_metadata` now receive `kv_manager` explicitly. This keeps logical KV state owned by Scheduler while still letting ModelRunner transform that state into model/backend inputs.

## Permanent Eviction And Sparse Compute
Permanent eviction remains a `KVManager`/Scheduler-side operation because it mutates request block tables and block reference counts. The external caller should route eviction through Scheduler's `kv_manager`.

Sparse compute remains a ModelRunner-side per-step metadata filter. It receives a temporary `SparseComputePlan`, skips specified logical block indices while building block tables and slot mappings, and does not mutate `KVManager`.

## Why This Design Is Better
Motivation: align KVCore with vLLM's logical/runtime split without reintroducing executor/worker complexity.

Expected gain: easier scheduler development, clearer ownership for future preemption/permanent eviction/sparse compute, and fewer hidden dependencies between model execution and block allocation.

Risk: ModelRunner APIs become slightly more verbose because they accept `kv_manager` explicitly.

Minimal validation plan:

- Unit test that Scheduler initializes and owns `KVManager`.
- Unit test that ModelRunner initializes KV tensors without owning `KVManager`.
- Unit test that metadata construction reads blocks from Scheduler-owned `KVManager`.
- Regression test sparse compute still filters per-step metadata without mutating logical KV state.

## Current Limits
This redesign still assumes single-process execution. It does not implement vLLM's executor, worker, async scheduling, KV cache profiling, or distributed KV cache config propagation.
