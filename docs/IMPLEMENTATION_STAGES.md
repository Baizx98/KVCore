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

When implementation reality changes, update this file.

---

## Phase 0: Skeleton

Goal:

- establish a minimal Python package layout
- define config and request state skeletons
- create a minimal engine entry path

Exit criteria:

- package imports work
- engine can be constructed
- a minimal decode smoke path exists

---

## Phase 1: KV metadata foundation

Goal:

- define block-level KV metadata
- introduce per-layer KV state
- build a basic allocator and sequence-to-block mapping

Exit criteria:

- block allocation and release are testable
- per-layer KV ownership is explicit
- request-specific KV views have a clear skeleton

---

## Phase 2: Layer-wise execution

Goal:

- convert execution into an explicit layer-by-layer runtime
- introduce batch and layer contexts
- make the control path KV-aware

Exit criteria:

- runtime iterates through layers explicitly
- attention no longer depends on opaque past-key-values style flow
- reference execution path is stable enough for comparison tests

---

## Phase 3: Scheduling and chunked prefill

Goal:

- add request progression logic
- support continuous batching
- support chunked prefill without bypassing KV abstractions

Exit criteria:

- request state transitions are explicit
- mixed prefill/decode scheduling is testable
- chunk progression preserves correct KV state

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

Exit criteria:

- residency state is explicit
- offload and reload preserve correctness
- bookkeeping is testable under repeated movement

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
