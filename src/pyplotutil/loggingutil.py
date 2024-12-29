"""Event logging utility for streamlined log management.

This module provides a robust logging system that combines file and console logging with
configurable verbosity levels. It features:

- Unified interface for file and console logging
- FakeLogger implementation for testing scenarios
- Automatic log file management
- Verbosity control through command line arguments

Classes
-------
FakeLogger
    Mock logger implementing standard logging interface

Functions
---------
start_event_logging
    Configure and initialize logging system
get_logging_level_from_verbose_count
    Convert verbosity flags to logging levels
start_logging
    Simplified logging setup with verbosity control
event_logger, evlog
    Convenient access to global logger instance

Examples
--------
>>> from pyplotutil.loggingutil import start_logging
>>> logger = start_event_logging(argv=["script.py"], output_dir="logs", logging_level="INFO")
>>> logger.info("Processing started")

Notes
-----
- Uses Python's built-in logging module as the foundation
- Automatically disables console output during pytest execution
- Supports both string and integer logging levels
- Maintains compatibility with standard logging interfaces

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
    Convert logging level string or int to corresponding integer level.

    Parameters
    ----------
    level : int or str
        Logging level as integer or string name (e.g. 'INFO', 'DEBUG')

    Returns
    -------
    int
        Integer logging level value

    Raises
    ------
    ValueError
        If level string is not recognized
    TypeError
        If level is not an integer or string

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
    Mock logger implementing standard logging interface.

    Provides logging functionality that mimics Python's standard Logger class, with output directed
    to a specified stream.

    Attributes
    ----------
    level : int
        Current logging level
    disabled : bool
        Whether logging is currently disabled
    terminator : str
        String appended after each log message

    """

    _level: int
    _formatter: logging.Formatter
    _stream: TextIO
    _disabled: bool
    terminator: str = "\n"

    def __init__(self, level: int | str = NOTSET, stream: TextIO | None = None, *, disable: bool = False) -> None:
        """
        Initialize FakeLogger with specified level and output stream.

        Parameters
        ----------
        level : int or str
            The logging level to set, either as an integer or a string (default is WARNING)
        stream : TextIO or None
            Output stream for log messages, defaults to sys.stderr
        disable : bool
            Whether to start with logging disabled (default is False)

        """
        self.set_level(level)
        self._formatter = _default_formatter
        self._stream = stream if stream is not None else sys.stderr
        self._disabled = disable

    def set_level(self, level: int | str) -> None:
        """
        Set the logging level threshold.

        Parameters
        ----------
        level : int or str
            New logging level to set

        """
        self._level = check_level(level)

    def setLevel(self, level: int | str) -> None:  # noqa: N802
        """
        Legacy-style alias for set_level.

        Parameters
        ----------
        level : int or str
            New logging level to set

        """
        return self.set_level(level)

    def is_enabled_for(self, level: int) -> bool:
        """
        Check if logger will process messages at given level.

        Parameters
        ----------
        level : int
            Logging level to check

        Returns
        -------
        bool
            True if messages at this level will be processed

        """
        if self.disabled:
            return False
        return level >= self.level

    def find_caller(self, *, stack_info: bool = False, stacklevel: int = 1) -> tuple[str, int, str, str | None]:
        """
        Identify the module and line number where logging was called.

        Parameters
        ----------
        stack_info : bool
            Whether to capture full stack information
        stacklevel : int
            Number of frames to skip when finding caller

        Returns
        -------
        tuple
            (filename, line number, function name, stack info)

        """
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
        Core logging method that handles message creation and routing.

        Parameters
        ----------
        level : int
            Logging level for this message
        msg : str
            Message format string
        args : tuple
            Arguments for message formatting
        **kwargs : dict
            Additional logging context

        """
        fn, lno, func, sinfo = self.find_caller(stack_info=stack_info, stacklevel=stacklevel)
        _ = extra
        record = logging.LogRecord("root", level, fn, lno, msg, args, exc_info, func, sinfo)
        self.handle(record)

    def handle(self, record: logging.LogRecord) -> None:
        """
        Process a LogRecord by emitting it.

        Parameters
        ----------
        record : logging.LogRecord
            The record to be processed

        """
        self.emit(record)

    def flush(self) -> None:
        """Ensure all logging output has been flushed to the stream."""
        if self._stream and hasattr(self._stream, "flush"):
            self._stream.flush()

    def emit(self, record: logging.LogRecord) -> None:
        """
        Write formatted LogRecord to output stream.

        Parameters
        ----------
        record : logging.LogRecord
            The record to be emitted

        """
        msg = self.format(record)
        self._stream.write(msg + self.terminator)
        self.flush()

    def set_formatter(self, fmt: logging.Formatter) -> None:
        """
        Set the formatter for log messages.

        Parameters
        ----------
        fmt : logging.Formatter
            Formatter to use for log messages

        """
        self._formatter = fmt

    def format(self, record: logging.LogRecord) -> str:
        """
        Apply formatter to LogRecord.

        Parameters
        ----------
        record : logging.LogRecord
            The record to format

        Returns
        -------
        str
            Formatted log message

        """
        return self._formatter.format(record)

    def debug(self, msg: str, *args: object, **kwargs: Unknown) -> None:
        """
        Log message at DEBUG level.

        Parameters
        ----------
        msg : str
            Message format string
        *args : tuple
            Arguments for message formatting
        **kwargs : dict
            Additional logging context

        """
        if self.is_enabled_for(DEBUG):
            self._log(DEBUG, msg, args, **kwargs)

    def info(self, msg: str, *args: object, **kwargs: Unknown) -> None:
        """
        Log message at INFO level.

        Parameters
        ----------
        msg : str
            Message format string
        *args : tuple
            Arguments for message formatting
        **kwargs : dict
            Additional logging context

        """
        if self.is_enabled_for(INFO):
            self._log(INFO, msg, args, **kwargs)

    def warning(self, msg: str, *args: object, **kwargs: Unknown) -> None:
        """
        Log message at WARNING level.

        Parameters
        ----------
        msg : str
            Message format string
        *args : tuple
            Arguments for message formatting
        **kwargs : dict
            Additional logging context

        """
        if self.is_enabled_for(WARNING):
            self._log(WARNING, msg, args, **kwargs)

    def error(self, msg: str, *args: object, **kwargs: Unknown) -> None:
        """
        Log message at ERROR level.

        Parameters
        ----------
        msg : str
            Message format string
        *args : tuple
            Arguments for message formatting
        **kwargs : dict
            Additional logging context

        """
        if self.is_enabled_for(ERROR):
            self._log(ERROR, msg, args, **kwargs)

    def critical(self, msg: str, *args: object, **kwargs: Unknown) -> None:
        """
        Log message at CRITICAL level.

        Parameters
        ----------
        msg : str
            Message format string
        *args : tuple
            Arguments for message formatting
        **kwargs : dict
            Additional logging context

        """
        if self.is_enabled_for(CRITICAL):
            self._log(CRITICAL, msg, args, **kwargs)

    def log(self, level: int, msg: str, *args: object, **kwargs: Unknown) -> None:
        """
        Log message at specified level.

        Parameters
        ----------
        level : int
            Logging level to use
        msg : str
            Message format string
        *args : tuple
            Arguments for message formatting
        **kwargs : dict
            Additional logging context

        """
        if self.is_enabled_for(level):
            self._log(level, msg, args, **kwargs)

    def toggle(self, *, disabled: bool | None = None) -> bool:
        """
        Toggle or set logger's disabled state.

        Parameters
        ----------
        disabled : bool or None
            New disabled state, or None to toggle current state

        Returns
        -------
        bool
            New disabled state

        """
        if disabled is None:
            self._disabled = not self._disabled
        else:
            self._disabled = disabled
        return self._disabled

    @property
    def level(self) -> int:
        """
        Get current logging level.

        Returns
        -------
        int
            Current logging level

        """
        return self._level

    @property
    def disabled(self) -> bool:
        """
        Get logger's disabled state.

        Returns
        -------
        bool
            Whether logging is currently disabled

        Return whether the console output is disabled or not.

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
    Configure and start event logging system.

    Sets up both console and file handlers with specified logging levels.

    Parameters
    ----------
    argv : list[str]
        Command line arguments used to generate default log filename
    output_dir : str or Path or None
        Directory for log file output
    log_filename : str or Path or None
        Custom log filename, generated from argv if None
    name : str or None
        Logger name, defaults to module name if None
    logging_level : int or str
        Console handler logging level
    logging_level_file : int or str
        File handler logging level
    fmt : str or None
        Custom log message format string

    Returns
    -------
    logging.Logger
        Configured logger instance ready for use

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
    Convert verbosity count to logging level string.

    Parameters
    ----------
    verbose_count : int
        Number of verbose flags (e.g. -v, -vv, -vvv)
        0: WARNING
        1: INFO
        2+: DEBUG

    Returns
    -------
    str
        Corresponding logging level name

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
    Initialize logging system with verbosity control and output configuration.

    Sets up a complete logging system with both file and console handlers based on the specified
    verbosity level. Creates output directory if it doesn't exist (unless in dry-run mode).

    Parameters
    ----------
    argv : list[str]
        Command line arguments used for log file naming
    output_dir : Path
        Directory where log files will be stored
    name : str
        Name for the logger instance
    verbose_count : int
        Verbosity level from command line:
        - 0: WARNING level
        - 1: INFO level
        - 2+: DEBUG level
    dry_run : bool, optional
        When True, prevents creation of output directory and log files,
        default is False

    Returns
    -------
    logging.Logger
        Configured logger instance ready for immediate use

    Examples
    --------
    >>> logger = start_logging(argv=["script.py"], output_dir=Path("logs"), name="myapp", verbose_count=1)
    >>> logger.info("Application started")

    """
    logging_level = get_logging_level_from_verbose_count(verbose_count)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    return start_event_logging(argv, output_dir, name=name, logging_level=logging_level)
