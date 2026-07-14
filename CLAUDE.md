# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Package manager is **uv**; run tools through it: `uv run pytest`, `uv run mypy src tests`, `uv run ruff check`.
- `nox` runs the default sessions (lint + tests across Python 3.10–3.13). Prefer `uv run pytest` for quick iteration.
- Run a single test: `uv run pytest -k 'test_name'`.
- Run `uv run mypy` before finishing a change (targets `src`, `tests`, and `examples` via config); CI enforces it through `nox -s typecheck`.

## Code style

- ruff has `select = ["ALL"]` (every rule enabled), line length 120, numpy docstring convention.
- Every module must start with `from __future__ import annotations` (ruff isort `required-imports`).
- Format with `ruff format` (Black-compatible, double quotes).

## Gotchas

- numpy is pinned `<2.2.0`. The pin may be revisitable, but never bump it without running the full test suite.
- Data handling is built on **polars**, not pandas.
- The package version comes from git tags via hatch-vcs — do not add a static `version` to pyproject.toml.
- pytest-randomly randomizes test order, so tests must not depend on execution order.

## Commits

- Conventional Commits with a capitalized, imperative subject: `feat: Add ...`, `fix: Handle ...`, `chore: Re-lock ...`.
