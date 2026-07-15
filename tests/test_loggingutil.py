# ruff: noqa: S101
"""Unit tests for the loggingutil module.

This test suite verifies the functionality of the utility functions, the `FakeLogger` class
including message emission, and the event logging setup helpers.

"""

from __future__ import annotations

import logging
import uuid
from io import StringIO
from pathlib import Path

import pytest

import pyplotutil.loggingutil
from pyplotutil.loggingutil import (
    CRITICAL,
    DEBUG,
    ERROR,
    FATAL,
    INFO,
    NOTSET,
    WARN,
    WARNING,
    FakeLogger,
    check_level,
    event_logger,
    evlog,
    get_event_logger_filename,
    get_logging_level_from_verbose_count,
    start_event_logging,
    start_logging,
)


@pytest.fixture
def isolated_event_logger(monkeypatch: pytest.MonkeyPatch) -> FakeLogger:
    """Replace the global event logger with a fresh FakeLogger for the duration of a test."""
    fake = FakeLogger(disable=True)
    monkeypatch.setattr(pyplotutil.loggingutil, "_event_logger", fake)
    return fake


def unique_logger_name() -> str:
    """Return a unique logger name so tests do not share logging.Logger instances."""
    return f"test_loggingutil_{uuid.uuid4().hex}"


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


class TestFakeLoggerEmission:
    """A class collecting tests for FakeLogger message emission."""

    @pytest.mark.parametrize(
        ("method", "level_name"),
        [
            ("debug", "DEBUG"),
            ("info", "INFO"),
            ("warning", "WARNING"),
            ("error", "ERROR"),
            ("critical", "CRITICAL"),
        ],
    )
    def test_emit_at_each_level(self, method: str, level_name: str) -> None:
        """Test that each level method writes a formatted message to the stream."""
        stream = StringIO()
        logger = FakeLogger(level=DEBUG, stream=stream)
        getattr(logger, method)("count is %d", 42)
        output = stream.getvalue()
        assert f"[{level_name}]: count is 42" in output
        assert output.endswith(logger.terminator)

    def test_log_with_explicit_level(self) -> None:
        """Test the generic log method with an explicit level."""
        stream = StringIO()
        logger = FakeLogger(level=DEBUG, stream=stream)
        logger.log(ERROR, "boom")
        assert "[ERROR]: boom" in stream.getvalue()

    def test_below_threshold_is_suppressed(self) -> None:
        """Test that messages below the level threshold are not written."""
        stream = StringIO()
        logger = FakeLogger(level=WARNING, stream=stream)
        logger.info("quiet")
        assert stream.getvalue() == ""

    def test_disabled_logger_writes_nothing(self) -> None:
        """Test that a disabled logger writes nothing at any level."""
        stream = StringIO()
        logger = FakeLogger(level=DEBUG, stream=stream, disable=True)
        logger.critical("silent")
        assert stream.getvalue() == ""

    def test_set_level_legacy_alias(self) -> None:
        """Test the camelCase setLevel alias."""
        logger = FakeLogger()
        logger.setLevel(ERROR)
        assert logger.level == ERROR

    def test_toggle(self) -> None:
        """Test toggling and explicitly setting the disabled state."""
        logger = FakeLogger()
        assert logger.toggle() is True
        assert logger.toggle() is False
        assert logger.toggle(disabled=True) is True
        assert logger.disabled

    def test_set_formatter(self) -> None:
        """Test that a custom formatter changes the output format."""
        stream = StringIO()
        logger = FakeLogger(level=DEBUG, stream=stream)
        logger.set_formatter(logging.Formatter("!%(message)s!"))
        logger.info("custom")
        assert stream.getvalue() == "!custom!" + logger.terminator

    def test_find_caller_reports_this_file(self) -> None:
        """Test that the caller lookup finds this test module."""
        logger = FakeLogger()
        filename, lineno, func, sinfo = logger.find_caller()
        assert Path(filename).name == "test_loggingutil.py" or filename.endswith("test_loggingutil.py")
        assert lineno > 0
        assert func == "test_find_caller_reports_this_file"
        assert sinfo is None

    def test_find_caller_with_stack_info(self) -> None:
        """Test that stack information is captured on request."""
        logger = FakeLogger()
        _, _, _, sinfo = logger.find_caller(stack_info=True)
        assert sinfo is not None
        assert sinfo.startswith("Stack (most recent call last):")


@pytest.mark.parametrize(
    ("verbose_count", "expected"),
    [(0, "WARNING"), (1, "INFO"), (2, "DEBUG"), (3, "DEBUG"), (10, "DEBUG")],
)
def test_get_logging_level_from_verbose_count(verbose_count: int, expected: str) -> None:
    """Test mapping of verbosity counts to logging level names."""
    assert get_logging_level_from_verbose_count(verbose_count) == expected


def test_event_logger_default_is_fake(isolated_event_logger: FakeLogger) -> None:
    """Test that the global event logger is a FakeLogger before logging is started."""
    assert event_logger() is isolated_event_logger
    assert evlog() is isolated_event_logger
    assert get_event_logger_filename() is None


@pytest.mark.usefixtures("isolated_event_logger")
def test_start_event_logging_creates_log_file(tmp_path: Path) -> None:
    """Test that starting event logging attaches a file handler and writes a log file."""
    name = unique_logger_name()
    logger = start_event_logging(["script.py"], output_dir=tmp_path, name=name)
    try:
        assert isinstance(logger, logging.Logger)
        assert event_logger() is logger
        log_filename = get_event_logger_filename()
        assert log_filename == tmp_path / "script.log"
        logger.debug("hello")
        assert log_filename.is_file()
        assert "hello" in log_filename.read_text()
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


@pytest.mark.usefixtures("isolated_event_logger")
def test_start_event_logging_is_idempotent(tmp_path: Path) -> None:
    """Test that a second start with the same name returns the same logger."""
    name = unique_logger_name()
    logger = start_event_logging(["script.py"], output_dir=tmp_path, name=name)
    try:
        assert start_event_logging(["script.py"], output_dir=tmp_path, name=name) is logger
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


@pytest.mark.usefixtures("isolated_event_logger")
def test_start_event_logging_warns_on_missing_directory(tmp_path: Path) -> None:
    """Test that a log path in a missing directory warns instead of raising."""
    name = unique_logger_name()
    missing = tmp_path / "does" / "not" / "exist" / "event.log"
    with pytest.warns(RuntimeWarning, match="Unable to save log file"):
        logger = start_event_logging(["script.py"], log_filename=missing, name=name)
    try:
        assert get_event_logger_filename() is None
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


@pytest.mark.usefixtures("isolated_event_logger")
def test_start_logging_creates_output_directory(tmp_path: Path) -> None:
    """Test that start_logging creates the output directory and sets console verbosity."""
    name = unique_logger_name()
    output_dir = tmp_path / "logs"
    logger = start_logging(["script.py"], output_dir, name, verbose_count=1)
    try:
        assert output_dir.is_dir()
        console_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert console_handlers[0].level == INFO
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


@pytest.mark.usefixtures("isolated_event_logger")
def test_start_logging_dry_run_skips_directory(tmp_path: Path) -> None:
    """Test that dry-run mode creates neither the directory nor the log file."""
    name = unique_logger_name()
    output_dir = tmp_path / "logs"
    with pytest.warns(RuntimeWarning, match="Unable to save log file"):
        logger = start_logging(["script.py"], output_dir, name, verbose_count=0, dry_run=True)
    try:
        assert not output_dir.exists()
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


# Local Variables:
# jinx-local-words: "FakeLogger StringIO argv camelCase evlog loggingutil monkeypatch noqa py setLevel uuid"
# End:
