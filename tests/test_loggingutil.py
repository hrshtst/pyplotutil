# ruff: noqa: S101
"""Unit tests for the loggingutil module.

This test suite verifies the functionality of the utility functions and `FakeLogger` class.

"""

from __future__ import annotations

import pytest

from pyplotutil.loggingutil import CRITICAL, DEBUG, ERROR, FATAL, INFO, NOTSET, WARN, WARNING, check_level


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (CRITICAL, CRITICAL),
        (FATAL, FATAL),
        (FATAL, CRITICAL),
        (ERROR, ERROR),
        (WARNING, WARNING),
        (WARN, WARN),
        (WARN, WARNING),
        (INFO, INFO),
        (DEBUG, DEBUG),
        (NOTSET, NOTSET),
        (42, 42),
    ],
)
def test_check_level_integer(level: int, expected: int) -> None:
    """Test if the logging level is returned when it is given as an integer."""
    assert check_level(level) == expected


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("CRITICAL", CRITICAL),
        ("FATAL", FATAL),
        ("FATAL", CRITICAL),
        ("ERROR", ERROR),
        ("WARNING", WARNING),
        ("WARN", WARN),
        ("WARN", WARNING),
        ("INFO", INFO),
        ("DEBUG", DEBUG),
        ("NOTSET", NOTSET),
    ],
)
def test_check_level_name(level: str, expected: int) -> None:
    """Test the logging level value when a valid level name is given."""
    assert check_level(level) == expected


def test_check_unknown_level_name() -> None:
    """Test if an exception is raised when unknown level name is given."""
    level = "unknown"
    msg = f"Unknown level: {level}"
    with pytest.raises(ValueError, match=msg):
        _ = check_level(level)
