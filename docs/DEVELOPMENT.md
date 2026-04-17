# Development

This document describes the recommended development workflow for the repository.

The project is intentionally small and focused.  
Tooling should stay simple and predictable.

---

## 1. Tooling

Use the following tools:

- `uv` for dependency and environment management
- `ruff` for linting and formatting
- `mypy` for gradual static type checking
- `pre-commit` for automated local checks
- `pytest` for testing

Optional later tools may be introduced when needed, but avoid unnecessary expansion.

`mypy` should start in a **loose configuration**.

Initial expectations:

- allow untyped functions and modules
- focus on catching obvious type errors early
- tighten rules gradually as the codebase stabilizes

---

## 2. Environment setup

### 2.1 Create environment

```bash
uv venv
source .venv/bin/activate
```

### 2.2 Install development dependencies

```bash
uv sync --group dev
```

### 2.3 Install pre-commit hooks

```bash
uv run pre-commit install
```

---

## 3. Commit message convention

Git commit messages should be short, explicit, and easy to scan.

Recommended format:

```text
<type>: <summary>
```

Rules:

- use lowercase type prefixes
- keep the summary concise and specific
- describe the actual change, not the intention to change
- one commit should represent one meaningful atomic change
- avoid mixing refactor, bugfix, docs, and formatting changes in a single commit when possible

Recommended types:

- `feat`: new feature or new user-visible capability
- `fix`: bug fix
- `refactor`: code restructuring without intended behavior change
- `doc`: documentation-only change
- `test`: add or modify tests
- `chore`: repository maintenance, tooling, dependency, or housekeeping updates
- `perf`: performance optimization
- `style`: formatting or non-functional style cleanup

Examples:

```text
feat: add minimal layer-wise engine skeleton
fix: correct kv block release on request finish
refactor: merge scheduler request and batch state
doc: rewrite architecture and implementation docs
test: add block pool lifecycle unit tests
chore: add pre-commit and mypy configuration
perf: reduce kv view rebuild overhead in decode path
style: normalize import ordering in runtime package
```

Notes:

- if the change is a real bug fix, use `fix` rather than `feat`
- use `doc` for documentation-only commits
- use `chore` for tooling or repository metadata changes such as `.gitignore`, `pyproject.toml`, hooks, or CI config

If a change is too broad to summarize cleanly with one type, it is usually a sign that the commit should be split.
