# Architecture

This document describes the **current architecture design target** for KVCore.

It is a design document, not a policy file.
If implementation reveals a better structure, this document should be updated.

---

## 1. Architectural intent

KVCore is designed around one central assumption:

> KV cache is a first-class runtime object.

The architecture should make the following explicit:

- layer-by-layer execution
- block-aware KV management
- separation of metadata and data
- hook points around attention
- room for hierarchical KV residency and selective KV use

---

## 2. High-level runtime flow

Target execution flow:

1. scheduler accepts and tracks requests
2. scheduler produces a batch plan
3. engine builds batch context
4. model execution proceeds layer by layer
5. pre-attention hooks update or prepare KV state
6. attention runs against block-aware KV views
7. post-attention hooks update residency, scores, or next-step metadata
8. logits are produced
9. scheduler commits results and updates request states

The system should avoid collapsing this into a single opaque whole-model forward when doing so would hide KV lifecycle control.

---

## 3. Core subsystem split

### 3.1 API

Responsibilities:

- configuration objects
- engine construction
- model loading entry points
- user-facing generation interface

This layer should stay thin.

### 3.2 Scheduler

Responsibilities:

- request lifecycle
- prefill/decode progression
- continuous batching
- chunked prefill planning
- finish conditions

The scheduler decides **what runs next**, not how KV is stored internally.

### 3.3 Runtime

Responsibilities:

- execute one step
- construct batch and layer contexts
- iterate through layers
- coordinate scheduler, model runtime, hooks, and KV subsystem
- later host CUDA-graph-safe decode execution if needed

This is the control plane of the system.

### 3.4 Model runtime

Responsibilities:

- adapt Qwen3 / Llama3 / Mistral3 into a unified execution interface
- hide model-family-specific naming and layout details
- expose layer-level execution boundaries needed by the runtime

### 3.5 KV subsystem

Responsibilities:

- block allocation
- layer-wise KV state tracking
- request-specific KV view construction
- residency tracking
- prefix reuse
- selection
- pruning
- offload coordination

This is the architectural center of the repository.

### 3.6 Hook subsystem

Responsibilities:

- pre-attention hook execution
- post-attention hook execution
- metadata updates around attention
- integration points for prefetch, offload, pruning, and scoring

### 3.7 Kernel/backend layer

Responsibilities:

- reference attention path
- paged attention path
- selective attention path
- memory movement helpers when needed

Recommended progression:

- reference implementation first
- optimized backend later

### 3.8 Metrics and benchmarks

Responsibilities:

- latency measurement
- throughput measurement
- memory accounting
- offload traffic accounting
- correctness and policy evaluation

---

## 4. Key data abstractions

The exact class names may change, but the design should preserve these concepts:

- `Request`
- `SequenceState`
- `BatchPlan`
- `BatchContext`
- `LayerContext`
- `KVBlock`
- `LayerKVState`
- `RequestKVView`
- `BlockAllocator`
- `PrefixCache`
- `OffloadCoordinator`
- `SelectionPolicy`
- `PrunePolicy`
- `HookManager`

Important semantic distinctions:

- `alive`: physically exists
- `eligible`: logically available for future use
- `selected`: participates in the current attention computation

These concepts should not be merged casually.

---

## 5. Directory scaffold target

The repository should gradually converge toward a structure similar to:

```text
kvcore/
  api/
  scheduler/
  runtime/
  model/
  kv/
  hooks/
  kernels/
  metrics/
  benchmark/
tests/
docs/
```

Suggested responsibilities:

- `kvcore/api/`
  - public construction and configuration interfaces
- `kvcore/scheduler/`
  - request state and batching logic
- `kvcore/runtime/`
  - step execution and layer orchestration
- `kvcore/model/`
  - model-family adapters
- `kvcore/kv/`
  - block metadata, allocation, views, offload, prefix reuse
- `kvcore/hooks/`
  - hook interfaces and implementations
- `kvcore/kernels/`
  - reference and optimized attention backends
- `kvcore/metrics/`
  - runtime metrics and measurement utilities
- `kvcore/benchmark/`
  - benchmark harnesses and experiment entry points
- `tests/`
  - unit, integration, and end-to-end tests

This scaffold is a target, not a frozen contract.
If implementation needs a cleaner arrangement, update this document accordingly.
