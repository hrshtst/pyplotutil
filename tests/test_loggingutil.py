# ruff: noqa: S101
"""Unit tests for the loggingutil module.

This test suite verifies the functionality of the utility functions and `FakeLogger` class.

"""

from __future__ import annotations

import pytest

from pyplotutil.loggingutil import CRITICAL, DEBUG, ERROR, FATAL, INFO, NOTSET, WARN, WARNING, FakeLogger, check_level


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


class TestFakeLogger:
    """A class collecting tests for `FakeLogger` class."""

    def test_init(self) -> None:
        """Test the initialization of FakeLogger class."""
        logger = FakeLogger()
        assert logger.level == NOTSET
        assert not logger.disabled

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
    def test_set_level_integer(self, level: int, expected: int) -> None:
        """Test setting logging level from a given integer value."""
        logger = FakeLogger()
        logger.set_level(level)
        assert logger.level == expected

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
    def test_set_level_name(self, level: str, expected: int) -> None:
        """Test setting logging level from a given level name."""
        logger = FakeLogger()
        logger.set_level(level)
        assert logger.level == expected

    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            (CRITICAL, True),
            (FATAL, True),
            (ERROR, False),
            (WARNING, False),
            (WARN, False),
            (INFO, False),
            (DEBUG, False),
            (NOTSET, False),
            (42, False),
            (13, False),
        ],
    )
    def test_is_enabled_for_critical(self, level: int, *, expected: bool) -> None:
        """Test if an inquiry level is enabled or not when CRITICAL is set."""
        logger = FakeLogger(level=CRITICAL)
        assert logger.is_enabled_for(level) is expected

    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            (CRITICAL, True),
            (FATAL, True),
            (ERROR, True),
            (WARNING, False),
            (WARN, False),
            (INFO, False),
            (DEBUG, False),
            (NOTSET, False),
            (42, True),
            (13, False),
        ],
    )
    def test_is_enabled_for_error(self, level: int, *, expected: bool) -> None:
        """Test if an inquiry level is enabled or not when ERROR is set."""
        logger = FakeLogger(level=ERROR)
        assert logger.is_enabled_for(level) is expected

    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            (CRITICAL, True),
            (FATAL, True),
            (ERROR, True),
            (WARNING, True),
            (WARN, True),
            (INFO, False),
            (DEBUG, False),
            (NOTSET, False),
            (42, True),
            (13, False),
        ],
    )
    def test_is_enabled_for_warning(self, level: int, *, expected: bool) -> None:
        """Test if an inquiry level is enabled or not when WARNING is set."""
        logger = FakeLogger(level=WARNING)
        assert logger.is_enabled_for(level) is expected

    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            (CRITICAL, True),
            (FATAL, True),
            (ERROR, True),
            (WARNING, True),
            (WARN, True),
            (INFO, True),
            (DEBUG, False),
            (NOTSET, False),
            (42, True),
            (13, False),
        ],
    )
    def test_is_enabled_for_info(self, level: int, *, expected: bool) -> None:
        """Test if an inquiry level is enabled or not when INFO is set."""
        logger = FakeLogger(level=INFO)
        assert logger.is_enabled_for(level) is expected

    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            (CRITICAL, True),
            (FATAL, True),
            (ERROR, True),
            (WARNING, True),
            (WARN, True),
            (INFO, True),
            (DEBUG, True),
            (NOTSET, False),
            (42, True),
            (13, True),
        ],
    )
    def test_is_enabled_for_debug(self, level: int, *, expected: bool) -> None:
        """Test if an inquiry level is enabled or not when DEBUG is set."""
        logger = FakeLogger(level=DEBUG)
        assert logger.is_enabled_for(level) is expected

    @pytest.mark.parametrize(
        "level",
        [CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET, 42, 13],
    )
    def test_is_enabled_when_disabled(self, level: int) -> None:
        """Test if return False always when the logger is disabled."""
        logger = FakeLogger(level=level, disable=True)
        assert not logger.is_enabled_for(CRITICAL)
        assert not logger.is_enabled_for(FATAL)
        assert not logger.is_enabled_for(ERROR)
        assert not logger.is_enabled_for(WARNING)
        assert not logger.is_enabled_for(WARN)
        assert not logger.is_enabled_for(INFO)
        assert not logger.is_enabled_for(DEBUG)
        assert not logger.is_enabled_for(NOTSET)
        assert not logger.is_enabled_for(42)
        assert not logger.is_enabled_for(13)


# Local Variables:
# jinx-local-words: "loggingutil noqa"
# End:
