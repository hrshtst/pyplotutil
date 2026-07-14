---
name: check
description: Run the full quality gate for pyplotutil — ruff lint, mypy type check, and the pytest suite. Use before committing or when the user asks to verify the codebase is green.
---

Run each step with Bash and report results. Continue through all steps even if an earlier one fails, so the user gets the complete picture in one pass.

1. **Lint**: `uv run ruff check`
2. **Format check**: `uv run ruff format --check`
3. **Type check**: `uv run mypy src tests`
4. **Tests**: `uv run pytest`

Then summarize: one line per step (pass/fail), followed by details for any failures.

- If ruff reports fixable issues, offer to run `uv run ruff check --fix` (note: `SIM105` is configured unfixable).
- mypy failures are expected to be fixed, not ignored — mypy is part of this repo's workflow even though CI does not run it.
- The test suite uses pytest-randomly; if a failure looks order-dependent, rerun with the printed seed (`-p randomly --randomly-seed=<seed>`) to reproduce.
