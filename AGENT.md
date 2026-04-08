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
