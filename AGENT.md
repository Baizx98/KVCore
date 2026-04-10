# AGENT.md

This file defines repository-level working rules for coding agents.

It is a policy file, not a design document.
Architecture details belong in `docs/ARCHITECTURE.md`.
Implementation planning belongs in `docs/IMPLEMENTATION_STAGES.md`.

---

## 1. Repository positioning

Treat this repository as a **research-oriented KV-centric LLM inference framework**.

Primary focus:

- KV cache as the core runtime abstraction
- layer-by-layer execution
- research-friendly iteration and refactoring

Do not treat this repository as a production serving stack.

---

## 2. Current engineering constraints

At the current stage, the project should remain **Python-first**.

Rules:

- prefer pure Python implementations
- keep tooling lightweight
- avoid introducing complex native build systems early
- if custom GPU kernels become necessary, prefer **Triton** first
- do not add C++/CUDA extension infrastructure unless explicitly justified by the user

This implies:

- no premature CMake setup
- no premature `pybind11` / custom extension build pipeline
- no architecture decisions that assume native extensions are already required

---

## 3. Scope constraints

Current supported model families:

- Qwen3
- Llama3
- Mistral3

Current intended feature scope:

- PagedAttention
- Prefix Cache
- Continuous Batching
- Chunked Prefill
- CPU Offload
- CUDA Graph
- selective block attention
- block-granularity pruning
- hook-based KV lifecycle control

Explicitly out of scope unless the user changes direction:

- beam search
- parallel sampling
- speculative decoding
- LoRA
- tensor / pipeline / expert parallelism
- distributed multi-node inference
- training or fine-tuning

---

## 4. Design and implementation rules

When writing or changing code:

- keep KV metadata and KV data conceptually separate
- use block/page granularity as the default management unit
- preserve layer-by-layer control flow
- keep scheduling policy separate from KV policy when practical
- prefer clarity, testability, and observability over premature optimization
- keep changes atomic and easy to review

Additional design guidance:

- borrow core ideas from recent vLLM architecture evolution, especially native KV offload design
- use `LLMEngine` as the top-level coordinator over API, scheduler, model runner, and KV management
- keep `ModelRunner` responsible for model loading, buffer initialization, profile runs, and model-executable metadata construction
- keep `KVManager` as the top-level KV owner, with `BlockPool` managing physical blocks
- move scheduler outputs toward flattened token-oriented batch metadata where practical
- keep prefill and decode as separate execution modes until chunked prefill is intentionally implemented
- treat data transfer as an independent abstraction rather than embedding transport details in attention logic
- prefer block-id-driven transfer specs over direct tensor-slice-oriented transfer APIs
- canonicalize model-family-specific KV layout into a shared KV view before offload, pruning, or selection logic consumes it
- model CPU->GPU and GPU->CPU movement as separate directional handlers rather than one generic move primitive
- design future offload around reusable pinned-memory CPU pools rather than temporary per-transfer buffers
- do not assume CPU and GPU block sizes must be identical; granularity differences should be modeled explicitly
- keep offload coordination aligned with layer execution and hook contexts rather than forcing a large connector framework too early
- avoid request-granular-only offload design; the long-term target should support request-, layer-, and block-granular movement, with block-granular control as the main path

If the current structure blocks clean implementation, refactoring is allowed.

---

## 5. Documentation policy

All repository documents are **living documents**:

- `AGENT.md`
- `README.md`
- files under `docs/`

Do not assume any existing document is permanently correct.

Required behavior:

- compare code, docs, and user intent continuously
- if implementation invalidates existing prose, update the relevant documents
- do not force code to match stale documentation
- `AGENT.md` itself may be revised when repository reality changes

Priority when conflicts appear:

1. latest explicit user instruction
2. cleaner and more maintainable implementation direction
3. existing repository documents

---

## 6. Development workflow expectations

Use the repository tooling unless the user asks otherwise:

- `uv`
- `ruff`
- `mypy` in loose mode initially
- `pytest`
- `pre-commit`

When adding checks or tooling:

- start with minimal friction
- prefer gradual tightening over strict-by-default policies
- avoid heavyweight infrastructure before the codebase stabilizes

Commit messages should follow the repository convention documented in `docs/DEVELOPMENT.md`.
Prefer one meaningful atomic change per commit.
