---
name: smart-git-commit
description: Use when the user asks to commit current git changes, such as "提交git", "git提交", "帮我提交", "commit these changes", or similar requests. Inspect unstaged, staged, and untracked changes in the current repository, split them into sensible atomic commits, and write commit messages that follow the repository convention from docs/DEVELOPMENT.md when available.
---

# Smart Git Commit

## Overview

This skill turns the current repository's uncommitted changes into one or more clean git commits.

It should:

- inspect staged, unstaged, and untracked changes
- group changes by intent into atomic commits
- skip obvious noise files by default unless they are intentionally tracked project artifacts
- follow the repository's commit message convention when documented
- write concise commit messages such as `feat: ...`, `fix: ...`, `doc: ...`, `refactor: ...`

## Workflow

### 1. Confirm repository context

First verify that the current directory is inside a git repository.

If not in a git repository:

- say so clearly
- do not create a repository
- do not run `git init`

### 2. Read the repository commit convention

Before composing commit messages:

- look for `docs/DEVELOPMENT.md`
- if it contains a commit message convention, follow it
- otherwise use the common `<type>: <summary>` format

Default commit types:

- `feat`
- `fix`
- `refactor`
- `doc`
- `test`
- `chore`
- `perf`
- `style`

Prefer repository-local rules over generic defaults.

### 3. Inspect all uncommitted changes

Inspect:

- staged changes
- unstaged changes
- untracked files

Use git status and diffs to understand:

- which files changed
- whether the changes are logically related
- whether multiple commits are needed
- whether a single file contains multiple unrelated edits

Before deciding commit groupings, filter out obvious noise.

Typical noise to skip by default:

- virtual environments such as `.venv/`, `venv/`
- Python caches such as `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- build or packaging outputs such as `build/`, `dist/`, `*.egg-info/`
- local editor or OS files such as `.DS_Store`, `.idea/`, `.vscode/`
- logs, temp files, coverage outputs, and ad hoc result dumps

Examples:

- `*.log`
- `*.tmp`
- `.coverage`
- `coverage.xml`
- `htmlcov/`

Do not auto-commit these files unless there is clear evidence they are intentionally tracked repository artifacts.

If a noise-like file is already tracked and modified, inspect it before skipping it.
Tracked changes may still be intentional and should not be discarded automatically.

### 4. Split changes by intent

The default goal is **good commit hygiene**, not "everything in one commit".

Group changes by meaningful intent, for example:

- feature work
- bug fixes
- documentation updates
- refactors without behavior change
- tests
- tooling or repository maintenance

Rules:

- one commit should represent one meaningful atomic change
- do not mix docs, tooling, bugfixes, and refactors in one commit unless they are tightly coupled
- if a file contains unrelated edits, use partial staging when practical
- prefer smaller coherent commits over one large mixed commit

Heuristics for splitting:

- docs-only edits usually become a `doc` commit
- `.gitignore`, hooks, formatter, linter, type checker, CI, and package metadata updates usually become a `chore` commit
- behavior-changing bug fixes usually become a `fix` commit
- new capabilities usually become a `feat` commit
- behavior-preserving structural cleanup usually becomes a `refactor` commit

### 5. Stage deliberately

Stage only the files or hunks that belong to the current commit.

Preferred approach:

- use normal staging when a file belongs wholly to one commit
- use partial staging when a file mixes unrelated changes

Do not stage unrelated work just because it is present in the tree.
Do not stage noise files by default.

### 6. Write the commit message

Use the repository convention if present.

Preferred format:

```text
<type>: <summary>
```

Message rules:

- choose the narrowest accurate type
- keep the summary short and concrete
- describe what changed, not what you plan to do
- avoid vague summaries like `update code` or `fix issues`

Examples:

- `feat: add minimal scheduler skeleton`
- `fix: preserve kv residency state during block release`
- `doc: clarify architecture and stage planning boundaries`
- `chore: add pre-commit and mypy configuration`

### 7. Commit sequentially

If the work needs multiple commits:

- create them one by one
- re-check status between commits
- continue until all intended changes are committed

At the end, report:

- how many commits were created
- each commit message
- any remaining uncommitted changes, if they still exist

## Decision rules

Ask the user before committing only when necessary, for example:

- the changes appear to contain multiple plausible split strategies with different meanings
- the repository contains risky or surprising unrelated edits that should not be auto-committed
- the only detected changes are ambiguous generated artifacts or tracked outputs
- the branch state suggests the user may expect a different workflow

Otherwise, make a reasonable split and proceed.

## Safety rules

- do not rewrite history unless explicitly requested
- do not amend existing commits unless explicitly requested
- do not revert user changes
- do not include generated noise or cache files unless they are intentionally tracked
- do not create empty commits unless explicitly requested

This skill is responsible for commit grouping and commit message generation.
Whether to run checks before committing depends on the repository workflow, for example whether `pre-commit` is configured, and is outside the scope of this skill.

## Expected output

After completing the task, briefly summarize:

- the commit grouping logic
- the commit messages created
- whether any changes were intentionally left uncommitted