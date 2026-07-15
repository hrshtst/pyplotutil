# ruff: noqa: S101
"""Unit tests for the package-level public API.

This test suite verifies that the curated public API is importable from the package root and that
the package exposes a version string.

"""

from __future__ import annotations

import pyplotutil


def test_all_names_resolve() -> None:
    """Test that every name in __all__ is an attribute of the package."""
    for name in pyplotutil.__all__:
        assert hasattr(pyplotutil, name)


def test_core_api_importable() -> None:
    """Test that the core classes and functions are importable from the package root."""
    from pyplotutil import Data, Dataset, TaggedData, plot_mean_err, save_figure, start_logging  # noqa: PLC0415

    assert callable(Data)
    assert callable(TaggedData)
    assert callable(Dataset)
    assert callable(save_figure)
    assert callable(plot_mean_err)
    assert callable(start_logging)


def test_version_is_set() -> None:
    """Test that the package exposes a non-empty version string."""
    assert isinstance(pyplotutil.__version__, str)
    assert pyplotutil.__version__ != ""


# Local Variables:
# jinx-local-words: "importable noqa pyplotutil str"
# End:
