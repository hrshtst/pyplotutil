"""Event Logging Utility Module.

This module provides utilities for event logging, leveraging the standard Python `logging` module to
streamline the creation of file and console handlers. It includes a `FakeLogger` class that mimics
the standard logger's methods, with logging messages directed to the standard output. Additionally,
helper functions simplify the setup of event logging and determine log verbosity based on user
preferences.

Notes
-----
- This module uses the Python `logging` module for robust log management.
- When `pytest` is detected, logging to the console is disabled by default to prevent unnecessary
  output during tests.

"""

from __future__ import annotations

import logging
import sys
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyplotutil._typing import Unknown

DEFAULT_FORMAT = "%(asctime)s (%(module)s:%(lineno)d) [%(levelname)s]: %(message)s"
_default_formatter = logging.Formatter(DEFAULT_FORMAT)

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

_LEVEL_TO_NAME = {
    CRITICAL: "CRITICAL",
    ERROR: "ERROR",
    WARNING: "WARNING",
    INFO: "INFO",
    DEBUG: "DEBUG",
    NOTSET: "NOTSET",
}
_NAME_TO_LEVEL = {
    "CRITICAL": CRITICAL,
    "FATAL": FATAL,
    "ERROR": ERROR,
    "WARN": WARNING,
    "WARNING": WARNING,
    "INFO": INFO,
    "DEBUG": DEBUG,
    "NOTSET": NOTSET,
}


def check_level(level: int | str) -> int:
    """
    Check validity of given level string and return an integer.

    Parameters
    ----------
    level : int or str
        An integer or a string representing a logging level

    Returns
    -------
    int
        The corresponding logging level as an integer.

    """
    if isinstance(level, int):
        rv = level
    elif str(level) == level:
        if level not in _NAME_TO_LEVEL:
            msg = f"Unknown level: {level}"
            raise ValueError(msg)
        rv = _NAME_TO_LEVEL[level]
    else:
        msg = f"Level not an integer or a valid string: {level}"
        raise TypeError(msg)
    return rv


class FakeLogger:
    """
    A fake logger class that mimics the behavior of a standard logger.

    Attributes
    ----------
    _console_output : bool
        Flag to determine whether console output is enabled.
    _logging_level : int
        The logging level currently set.
    _logging_level_map : ClassVar[dict[str, int]]
        A mapping of logging level names to their corresponding integer values.

    """

    _level: int
    _formatter: logging.Formatter
    _stream: TextIO
    _disabled: bool
    terminator: str = "\n"

    def __init__(self, level: int | str = NOTSET, stream: TextIO | None = None, *, disable: bool = False) -> None:
        """
        Initialize the FakeLogger.

        Parameters
        ----------
        level : int or str, optional
            The logging level to set, either as an integer or a string (default is WARNING).
        stream : TextIO, optional
            The stream in which logging output will be emitted (default is sys.stderr).
        disable : bool, optional
            Whether to disable console output (default is False).

        """
        self.set_level(level)
        self._formatter = _default_formatter
        self._stream = stream if stream is not None else sys.stderr
        self._disabled = disable

    def set_level(self, level: int | str) -> None:
        """
        Set the logging level.

        Parameters
        ----------
        level : int or str
            The logging level to set, either as an integer or a string.

        """
        self._level = check_level(level)

    def setLevel(self, level: int | str) -> None:  # noqa: N802
        """
        Alias for set_level.

        Parameters
        ----------
        level : int or str
            The logging level to set, either as an integer or a string.

        """
        return self.set_level(level)

    def is_enabled_for(self, level: int) -> bool:
        """Check if this logger is enabled for a level `level`.

        Parameters
        ----------
        level : int
            The logging level to check if this logger is enabled.

        """
        if self.disabled:
            return False
        return level >= self.level

    def find_caller(self, *, stack_info: bool = False, stacklevel: int = 1) -> tuple[str, int, str, str | None]:
        """Find the stack frame of the caller including the source file name, line number and function name."""
        _ = stack_info, stacklevel
        return "(unknown file)", 0, "(unknown function)", None

    def _log(
        self,
        level: int,
        msg: str,
        args: tuple[object, ...],
        *,
        exc_info: tuple[None, None, None] | None = None,
        extra: Mapping[str, object] | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:
        """
        Low-level logging routine.

        Parameters
        ----------
        level : int
            The logging level for the message.
        msg : str
            The message to log.
        *args : object
            Arguments to format the message.
        stacklevel : int, optional
            The stack level for the log (default is 1).

        """
        fn, lno, func, sinfo = self.find_caller(stack_info=stack_info, stacklevel=stacklevel)
        _ = extra
        record = logging.LogRecord("root", level, fn, lno, msg, args, exc_info, func, sinfo)
        self.handle(record)

    def handle(self, record: logging.LogRecord) -> None:
        """Handle a record."""
        self.emit(record)

    def flush(self) -> None:
        """Ensure logging output has been flushed."""
        if self._stream and hasattr(self._stream, "flush"):
            self._stream.flush()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record formatted by the specified formatter."""
        msg = self.format(record)
        self._stream.write(msg + self.terminator)
        self.flush()

    def set_formatter(self, fmt: logging.Formatter) -> None:
        """Set the formatter for this fake handler."""
        self._formatter = fmt

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record."""
        return self._formatter.format(record)

    def debug(self, msg: str, *args: object, **kwargs: Unknown) -> None:
        """
        Log a debug-level message.

        Parameters
        ----------
        msg : str
            The debug message.
        *args : object
            Arguments to format the message.

        """
        if self.is_enabled_for(DEBUG):
            self._log(DEBUG, msg, args, **kwargs)

    def info(self, msg: str, *args: object, **kwargs: Unknown) -> None:
        """
        Log a info-level message.

        Parameters
        ----------
        msg : str
            The info message.
        *args : object
            Arguments to format the message.

        """
        if self.is_enabled_for(INFO):
            self._log(INFO, msg, args, **kwargs)

    def warning(self, msg: str, *args: object, **kwargs: Unknown) -> None:
        """
        Log a warning-level message.

        Parameters
        ----------
        msg : str
            The warning message.
        *args : object
            Arguments to format the message.

        """
        if self.is_enabled_for(WARNING):
            self._log(WARNING, msg, args, **kwargs)

    def error(self, msg: str, *args: object, **kwargs: Unknown) -> None:
        """
        Log a error-level message.

        Parameters
        ----------
        msg : str
            The error message.
        *args : object
            Arguments to format the message.

        """
        if self.is_enabled_for(ERROR):
            self._log(ERROR, msg, args, **kwargs)

    def critical(self, msg: str, *args: object, **kwargs: Unknown) -> None:
        """
        Log a critical-level message.

        Parameters
        ----------
        msg : str
            The critical message.
        *args : object
            Arguments to format the message.

        """
        if self.is_enabled_for(CRITICAL):
            self._log(CRITICAL, msg, args, **kwargs)

    def log(self, level: int, msg: str, *args: object, **kwargs: Unknown) -> None:
        """
        Log a message with the integer severity `level`.

        Parameters
        ----------
        level : int
            The logging severity.
        msg : str
            The debug message.
        *args : object
            Arguments to format the message.

        """
        if self.is_enabled_for(level):
            self._log(level, msg, args, **kwargs)

    def toggle(self, *, disabled: bool | None = None) -> bool:
        """
        Toggle the output of the fake logger.

        Parameters
        ----------
        disabled : bool, optional
            Set the status of the fake logger.
        """
        if disabled is None:
            self._disabled = not self._disabled
        else:
            self._disabled = disabled
        return self._disabled

    @property
    def level(self) -> int:
        """
        Get the current logging level.

        Returns
        -------
        int
            The current logging level.

        """
        return self._level

    @property
    def disabled(self) -> bool:
        """
        Return whether the console output is disabled or not.

        Returns
        -------
        bool
            Whether the console output is disabled or not.

        """
        return self._disabled


def _running_in_pytest() -> bool:
    """
    Check if the code is running inside pytest.

    Returns
    -------
    bool
        True if running in pytest, False otherwise.

    """
    return "pytest" in sys.modules


_event_logger: logging.Logger | FakeLogger = FakeLogger(disable=_running_in_pytest())


def _get_default_event_log_filename(
    argv: list[str],
    output_dir: str | Path | None,
    given_log_filename: str | Path | None,
) -> Path:
    """
    Get the default event log filename.

    Parameters
    ----------
    argv : list of str
        Command-line arguments.
    output_dir : str or Path or None
        Directory to save the log file, or None for the default temporary directory.
    given_log_filename : str or Path or None
        Specific log filename, or None to generate one.

    Returns
    -------
    Path
        The path to the log file.

    """
    event_log_filename: Path
    if given_log_filename is None:
        if output_dir is not None:
            event_log_filename = (Path(output_dir) / Path(argv[0]).name).with_suffix(".log")
        else:
            event_log_filename = (Path(tempfile.gettempdir()) / Path(argv[0]).name).with_suffix(".log")
    else:
        event_log_filename = Path(given_log_filename)

    return event_log_filename


def start_event_logging(
    argv: list[str],
    output_dir: str | Path | None = None,
    log_filename: str | Path | None = None,
    name: str | None = None,
    logging_level: int | str = logging.WARNING,
    logging_level_file: int | str = logging.DEBUG,
    fmt: str | None = None,
) -> logging.Logger:
    """
    Start event logging.

    Parameters
    ----------
    argv : list of str
        Command-line arguments.
    output_dir : str or Path or None, optional
        Directory to save the log file (default is None).
    log_filename : str or Path or None, optional
        Specific log filename (default is None).
    name : str or None, optional
        Logger name (default is None).
    logging_level : int or str, optional
        Console logging level (default is logging.WARNING).
    logging_level_file : int or str, optional
        File logging level (default is logging.DEBUG).
    fmt : str or None, optional
        Log message format (default is None).

    Returns
    -------
    logging.Logger
        The configured logger instance.

    """
    if fmt is None:
        fmt = DEFAULT_FORMAT
    if name is None:
        name = __name__

    global _event_logger  # noqa: PLW0603
    if isinstance(_event_logger, logging.Logger) and _event_logger.name == name:
        # Event logging has been started.
        return _event_logger

    # Create a new logger instance.
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter(fmt)

    # Setup a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Setup a file handler
    event_log_filename = _get_default_event_log_filename(argv, output_dir, log_filename)
    try:
        fh = logging.FileHandler(event_log_filename)
    except FileNotFoundError:
        # Maybe, running in dry-run mode...
        msg = f"Unable to save log file (if running in dry-run mode, ignore this): {event_log_filename}"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    else:
        fh.setLevel(logging_level_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Always log command arguments.
    logger.debug("Start event logging")
    logger.debug("Logger name: %s", logger.name)
    logger.debug("Log filename: %s", event_log_filename)
    cmd = "python " + " ".join(argv)
    logger.debug("Command: %s", cmd)

    _event_logger = logger
    return _event_logger


def event_logger() -> logging.Logger | FakeLogger:
    """
    Get the global event logger instance.

    Returns
    -------
    logging.Logger or FakeLogger
        The global event logger instance.

    """
    return _event_logger


def evlog() -> logging.Logger | FakeLogger:
    """
    Alias for getting the global event logger instance.

    Returns
    -------
    logging.Logger or FakeLogger
        The global event logger instance.

    """
    return event_logger()


def get_logging_level_from_verbose_count(verbose_count: int) -> str:
    """
    Get the logging level based on the verbose count.

    Parameters
    ----------
    verbose_count : int
        The verbosity level (0 for WARNING, 1 for INFO, 2 or more for DEBUG).

    Returns
    -------
    str
        The corresponding logging level as a string.

    """
    match verbose_count:
        case 0:
            return "WARNING"
        case 1:
            return "INFO"
        case _:
            return "DEBUG"


def start_logging(
    argv: list[str],
    output_dir: Path,
    name: str,
    verbose_count: int,
    *,
    dry_run: bool = False,
) -> logging.Logger:
    """
    Start logging with verbosity control.

    Parameters
    ----------
    argv : list of str
        Command-line arguments.
    output_dir : Path
        Directory to save the log file.
    name : str
        Logger name.
    verbose_count : int
        Verbosity level (0 for WARNING, 1 for INFO, 2 or more for DEBUG).
    dry_run : bool, optional
        If True, do not actually write logs (default is False).

    Returns
    -------
    logging.Logger
        The configured logger instance.

    """
    logging_level = get_logging_level_from_verbose_count(verbose_count)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    return start_event_logging(argv, output_dir, name=name, logging_level=logging_level)
