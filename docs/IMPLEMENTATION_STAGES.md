# Implementation Stages

This document is the **current staged implementation plan** for KVCore.

It is expected to change.
As the repository evolves, phases may be split, merged, reordered, or rewritten.

Do not treat this file as a fixed contract.

---

## Planning rules

The roadmap should follow these principles:

- correctness before optimization
- reference path before optimized path
- explicit KV abstractions before advanced policies
- stable tests before performance claims
- use first-stage implementations to preserve clean evolution toward layer-wise control and block-oriented offload
- learn from recent vLLM offload design where useful, especially transport abstraction, canonical KV views, pinned CPU pools, and explicit transfer granularity
- do not copy vLLM's connector stack too early; keep KVCore aligned with its own layer-context and hook-driven execution model
- keep `LLMEngine` as the top-level coordinator over scheduler, model runner, and KV manager
- move toward flattened token-oriented batch metadata, while keeping prefill and decode separate until chunked prefill is intentionally introduced

When implementation reality changes, update this file.

---

## Phase 0: Skeleton

Goal:

- establish a minimal Python package layout
- define config and request state skeletons
- create a minimal engine entry path
- load a Hugging Face Llama checkpoint through reusable model-adapter code
- run a minimal single-request greedy decode smoke path

Exit criteria:

- package imports work
- engine can be constructed
- Llama tokenizer/model loading works through KVCore entry points
- a minimal greedy decode smoke path exists

---

## Phase 1: KV metadata foundation

Goal:

- define block-level KV metadata
- define canonical KV references that hide model-family-specific layout details
- introduce per-layer KV state
- build a basic allocator and sequence-to-block mapping
- keep metadata separate from the actual `transformers` `past_key_values` storage used in the initial reference path

Exit criteria:

- block allocation and release are testable
- per-layer KV ownership is explicit
- request-specific KV views have a clear skeleton
- canonical KV references are explicit enough to support future offload and selection work without redesign

---

## Phase 2: Layer-wise execution

Goal:

- introduce a top-level `LLMEngine` / `Scheduler` / `ModelRunner` / `KVManager` split
- convert execution into an explicit layer-by-layer runner path
- introduce batch and layer contexts plus complete attention metadata
- make the control path KV-aware under a continuous-KV assumption

Exit criteria:

- `LLMEngine` is the top-level coordinator
- scheduler exposes explicit prefill or decode batches
- `ModelRunner` can execute prefill and decode separately
- runtime iterates through layers explicitly
- attention no longer depends on opaque past-key-values style flow at the engine boundary
- reference execution path is stable enough for comparison tests

---

## Phase 3: Scheduling and chunked prefill

Goal:

- add request progression logic
- support continuous batching
- support chunked prefill without bypassing KV abstractions
- move scheduler batch outputs toward flattened token-oriented metadata

Exit criteria:

- request state transitions are explicit
- mixed prefill/decode scheduling is testable
- chunk progression preserves correct KV state
- flattened batch metadata remains consistent with request-level bookkeeping

---

## Phase 4: Prefix reuse

Goal:

- support block-level contiguous prefix reuse
- integrate prefix hits into scheduler/runtime behavior

Exit criteria:

- exact and partial prefix-hit behavior is testable
- model/config mismatches do not create false reuse

---

## Phase 5: CPU offload

Goal:

- add hierarchical KV residency
- support controlled CPU/GPU block movement
- integrate offload with the layer-wise control path
- use a transport abstraction that separates policy, orchestration, and directional transfer handlers
- use reusable pinned CPU pools rather than per-transfer temporary buffers
- allow CPU and GPU block granularities to differ when beneficial

Exit criteria:

- residency state is explicit
- offload and reload preserve correctness
- bookkeeping is testable under repeated movement
- transfer APIs are block-spec-driven rather than tensor-slice-driven
- request-, layer-, and block-granular movement paths are representable, with block-granular control as the target path

---

## Phase 6: Hook integration

Goal:

- make pre-attention and post-attention lifecycle control explicit
- integrate prefetch, offload, score updates, and related metadata updates

Exit criteria:

- hook order is stable
- metadata changes through hooks are observable and testable

---

## Phase 7: Selective KV participation

Goal:

- allow attention to use only selected KV blocks
- keep selection semantics distinct from pruning semantics

Exit criteria:

- full-path equivalence holds when selection is disabled
- selected-block views are testable and explicit

---

## Phase 8: Pruning

Goal:

- support logical and physical KV reduction
- preserve the distinction between `alive`, `eligible`, and `selected`

Exit criteria:

- pruning semantics are explicit
- interactions with selection, prefix reuse, and offload are testable

---

## Phase 9: Optimized kernels and decode optimization

Goal:

- add optimized attention backends where justified
- prefer Triton-first optimization paths
- later add controlled decode-path optimizations such as CUDA graph support

Exit criteria:

- optimized path matches reference-path correctness
- optimization remains optional rather than architecture-defining

---

## Phase 10: Benchmarks and evaluation

Goal:

- make the system measurable for research iteration
- add reproducible latency, throughput, and memory evaluation paths

Exit criteria:

- benchmark scripts are runnable
- baseline comparisons are reproducible
- KV policy tradeoffs can be measured consistently
