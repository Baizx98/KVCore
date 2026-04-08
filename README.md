# KVCore

KVCore is a research-oriented LLM inference framework centered on **KV cache management**.

The project is intentionally narrow in scope.  
It is used to explore runtime design questions around:

- paged KV management
- prefix reuse
- continuous batching
- chunked prefill
- CPU offload
- selective KV participation
- KV lifecycle hooks

The repository is still in an exploratory stage.  
Architecture, interfaces, and documentation may all change as implementation evolves.

---

## Project direction

KVCore treats KV cache as a first-class runtime-managed object rather than a side effect of attention.

Current design direction:

- layer-by-layer execution
- block-aware KV metadata
- explicit pre/post attention hook points
- research-friendly, easy-to-refactor code structure

Current implementation constraint:

- keep the project Python-first
- prefer Triton over C++/CUDA extensions if custom kernels are needed later

---

## Current model scope

- Qwen3
- Llama3
- Mistral3

Only decoder-only inference is in scope.

---

## Repository guide

- `AGENT.md`
  - repository working rules for coding agents
- `docs/ARCHITECTURE.md`
  - current architecture design and directory scaffold
- `docs/IMPLEMENTATION_STAGES.md`
  - dynamic implementation roadmap
- `docs/DEVELOPMENT.md`
  - development environment and tooling

---

## Status

This repository is not yet a stable framework release.
It should be treated as an evolving research codebase.
