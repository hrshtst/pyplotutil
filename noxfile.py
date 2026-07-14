# ruff: noqa: D100
from __future__ import annotations

from pathlib import Path

import nox

nox.needs_version = ">=2026.7.11"
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["lint", "typecheck", "tests"]

PYPROJECT = nox.project.load_toml("pyproject.toml")

ALL_PYTHONS = [
    c.split()[-1] for c in PYPROJECT["project"]["classifiers"] if c.startswith("Programming Language :: Python :: 3.")
]


@nox.session(python="3.12", reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run ruff linting."""
    session.install("ruff")
    session.run("ruff", "check", *session.posargs)


@nox.session(python="3.12", reuse_venv=True)
def typecheck(session: nox.Session) -> None:
    """Run mypy type checking."""
    session.install(*PYPROJECT["dependency-groups"]["dev"], "uv")
    session.install("-e.")
    # Point mypy at the session venv instead of the .venv hardcoded in pyproject.toml.
    session.run("mypy", f"--python-executable={Path(session.bin) / 'python'}", *session.posargs)


@nox.session(python=ALL_PYTHONS, reuse_venv=True)
def tests(session: nox.Session) -> None:
    """Run test suite with pytest."""
    session.install(*PYPROJECT["dependency-groups"]["dev"], "uv")
    session.install("-e.")
    session.run("pytest", *session.posargs)


# Local Variables:
# jinx-local-words: "dev mypy noqa pyproject pytest uv venv virtualenv"
# End:
