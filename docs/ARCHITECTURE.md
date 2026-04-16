# Architecture

This document describes the **current architecture design target** for KVCore.

It is a design document, not a policy file.
If implementation reveals a better structure, this document should be updated.

---

## 1. Architectural intent

KVCore is a research-oriented LLM inference framework centered on one assumption:

> KV cache is a first-class runtime object managed by the engine.

The current design target is:

- a thin API layer
- one top-level `LLMEngine` that coordinates the system
- a minimal scheduler that manages request progression
- a `ModelRunner` that hides model-family details and executes the model
- a `KVManager` that owns KV cache metadata and physical block resources
- explicit layer-wise execution boundaries
- batch representations that can scale toward flattened token scheduling

Reference direction:

- learn from recent vLLM design evolution where it helps, especially block pools, token-flattened batching, KV cache manager layering, and offload abstractions
- do not copy vLLM's entire connector or serving stack mechanically
- keep KVCore aligned with its own layer-wise and research-friendly control flow

---

## 2. High-level runtime flow

Target execution flow:

1. API receives one or more generation requests
2. API forwards requests to `LLMEngine`
3. `LLMEngine` inserts requests into the scheduler
4. scheduler selects either a prefill batch or a decode batch
5. `KVManager` prepares block metadata and physical block assignments
6. `ModelRunner` converts the scheduled batch into model input metadata
7. model execution proceeds layer by layer
8. `LLMEngine` consumes outputs, updates request states, and releases finished KV blocks
9. API returns completed results

Current simplifications:

- prefill and decode are separate execution modes
- chunked prefill is not implemented yet
- offload is not implemented yet
- only decoder-only inference is in scope
- KVCore intentionally skips a vLLM-style `KVCacheCoordinator` for now; each layer owns one
  `SingleTypeKVManager`, and the top-level `KVManager` coordinates those managers directly

---

## 3. Core subsystem split

### 3.1 API

Responsibilities:

- user-facing request and response objects
- engine construction
- generation entry points
- returning final inference results

This layer should stay thin.

### 3.2 LLMEngine

Responsibilities:

- act as the top-level coordinator
- own `Scheduler`, `KVManager`, and `ModelRunner`
- accept requests from the API
- drive engine steps until requests finish
- update request lifecycle state
- return final outputs to the API layer

`LLMEngine` should orchestrate, not own model-family logic or low-level KV policy details.

### 3.3 Scheduler

Responsibilities:

- maintain `waiting` and `running` queues
- decide which requests move into execution
- produce either a prefill batch or a decode batch
- expose batch metadata in a form the runner can consume

Current design constraints:

- only `waiting` and `running` queues are maintained
- no chunked prefill
- no preemption
- no offload-aware scheduling yet

Batch design direction:

- scheduler output should move toward flattened token-oriented batch metadata
- request-level bookkeeping remains explicit even if execution is token-flattened

### 3.4 ModelRunner

Responsibilities:

- hide model-family-specific details
- load model, tokenizer, and relevant configs
- initialize KV cache bottom-level buffers
- perform profile runs
- transform scheduled batches into model-executable metadata
- execute prefill or decode
- move model outputs back into engine-owned state objects

Current model focus:

- Llama first
- Qwen and Mistral later

`ModelRunner` is the execution center for model-specific behavior.

### 3.5 KV subsystem

Top-level owner:

- `KVManager`

Responsibilities:

- own KV cache lifecycle
- manage request-to-block mappings
- expose per-layer KV views
- provide attention metadata derived from block mappings
- coordinate physical block allocation through a block pool
- later integrate offload

Suggested internal structure:

- `KVManager`
- `SingleTypeKVManager`
- `BlockPool`
- `BlockTable`
- `RequestBlockTable`
- `LayerKVState`
- `RequestKVView`
- later `OffloadManager`

Design notes:

- `KVManager` is the architectural center of the repository
- per-layer state should remain explicit
- the current simplified target is one `SingleTypeKVManager` per layer
- `KVManager` is the direct owner of all layer managers; there is no coordinator layer in KVCore
- model-family-specific layout should be canonicalized before KV policy logic consumes it

The near-term KV stack is intentionally narrower than vLLM:

1. `KVManager` is the request-facing logical control plane.
2. `SingleTypeKVManager` owns the block table for exactly one attention layer.
3. `BlockPool` owns physical block metadata, free ordering, and prefix-cache hash metadata.
4. `BlockTable` maps KV-manager block ids to kernel-consumable block ids at execution time.

This keeps the code small while preserving the control points needed for prefix caching,
layer-aware policies, offload, and future paged attention kernels.

### 3.6 BlockPool

`BlockPool` should follow the same high-level philosophy as vLLM:

- blocks are pre-created and reused
- physical blocks are managed through a free structure rather than ad hoc allocation
- block identity is stable and reusable
- future prefix reuse and eviction should be able to build on the same block pool

Current role:

- manage physical block ownership
- allocate and free blocks
- support request lifecycle transitions
- retain cached block hash metadata without forcing physical block deduplication

Future role:

- support cached blocks
- support prefix hits
- support offload-aware residency

### 3.7 Hook subsystem

Long-term responsibilities:

- pre-attention hook execution
- post-attention hook execution
- prefetch, offload, pruning, and score-update integration

Current design rule:

- main execution control should stay in the runner and engine
- hooks are extension points around explicit layer boundaries, not the primary control mechanism
- Llama-family adapters expose KVCore attention metadata by installing per-layer hook state on
  the Hugging Face `self_attn` module before attention executes and removing it immediately after

### 3.8 Kernel/backend layer

Responsibilities:

- reference attention path
- future paged attention path
- future selective attention path
- later memory-movement helpers

Recommended progression:

- reference implementation first
- optimized backend later

---

## 4. Key runtime objects

The exact class names may change, but the design should preserve these concepts:

- `GenerateRequest`
- `GenerateResult`
- `RequestState`
- `LLMEngine`
- `Scheduler`
- `ScheduledBatch`
- `BatchContext`
- `LayerContext`
- `ModelRunner`
- `ModelAdapter`
- `KVManager`
- `SingleTypeKVManager`
- `BlockPool`
- `KVBlock`
- `LayerKVState`
- `RequestKVView`
- `AttentionMetadata`
- later `OffloadManager`

Important semantic distinctions:

- `alive`: physically exists
- `eligible`: logically available for future use
- `selected`: participates in the current attention computation

These concepts should not be merged casually.

---

## 5. Batch and attention metadata

Batch design direction should follow these rules:

- scheduler outputs should be compatible with flattened token-oriented execution
- request-level bookkeeping remains explicit even if tokens are flattened
- prefill and decode remain separate execution modes for now

Suggested `ScheduledBatch` fields:

- `mode`
- `request_ids`
- `num_requests`
- `num_tokens`
- `flat_input_ids`
- `flat_position_ids`
- `request_offsets`
- `request_token_counts`

Suggested `AttentionMetadata` fields:

- `mode`
- `layer_id`
- `num_requests`
- `num_tokens`
- `request_offsets`
- `request_token_counts`
- `slot_mapping`
- `block_tables`
- `context_lens`
- `query_start_locs`
- `seq_lens`
- `max_seq_len`
- `max_query_len`
- `kv_cache_dtype`

Current simplification:

- the physical KV layout may be continuous and sufficiently large
- metadata should still be designed as if future paged/block execution will consume it
- the reference Llama path still uses Hugging Face cache storage for numerical correctness, while
  KVCore passes explicit block metadata to attention hooks for future kernel replacement

---

## 6. KV layout progression

Near-term design:

- continuous KV cache buffers
- enough capacity for current requests
- explicit layer-wise state
- explicit block metadata even when physical storage is contiguous

This gives a clean transition path:

1. continuous physical buffers
2. explicit block metadata and request block tables
3. paged/block attention backends
4. offload, selection, and pruning on top of the same metadata

The important rule is:

- physical simplicity is acceptable early
- interface simplicity that blocks later research is not

---

## 7. Directory scaffold target

The repository should gradually converge toward a structure similar to:

```text
kvcore/
  api/
  engine/
  scheduler/
  model_runner/
  model/
  kv/
  runtime/
  hooks/
  kernels/
  metrics/
  benchmark/
tests/
docs/
example/
```

Suggested responsibilities:

- `kvcore/api/`
  - public request, response, config, and construction interfaces
- `kvcore/engine/`
  - `LLMEngine` and top-level coordination logic
- `kvcore/scheduler/`
  - request queues and batch planning
- `kvcore/model_runner/`
  - model execution entry points, metadata construction, and buffer initialization
- `kvcore/model/`
  - model-family adapters
- `kvcore/kv/`
  - block pool, block metadata, request views, and future offload
- `kvcore/runtime/`
  - execution contexts and shared runtime data objects
- `kvcore/hooks/`
  - optional pre/post layer lifecycle extensions
- `kvcore/kernels/`
  - reference and optimized attention backends
- `kvcore/metrics/`
  - runtime metrics and measurement utilities
- `kvcore/benchmark/`
  - benchmark harnesses and experiment entry points

This scaffold is a target, not a frozen contract.
If implementation needs a cleaner arrangement, update this document accordingly.
