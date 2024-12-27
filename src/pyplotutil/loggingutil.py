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
from typing import ClassVar

DEFAULT_LOG_FORMATTER = "%(asctime)s (%(module)s:%(lineno)d) [%(levelname)s]: %(message)s"


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

    _console_output: bool
    _logging_level: int
    _logging_level_map: ClassVar[dict[str, int]] = {
        "NOTSET": 0,
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }

    def __init__(self, *, disable_console_output: bool = False) -> None:
        """
        Initialize the FakeLogger.

        Parameters
        ----------
        disable_console_output : bool, optional
            Whether to disable console output (default is False).
        """
        self._disable_console_output = disable_console_output
        self._logging_level = logging.WARNING

    def _format(self, msg: str, *args: object) -> str:
        """
        Format a message string with the provided arguments.

        Parameters
        ----------
        msg : str
            The message template.
        *args : object
            Arguments to format the message with.

        Returns
        -------
        str
            The formatted message.
        """
        formatted_msg = msg % args
        if not formatted_msg.endswith("\n"):
            formatted_msg += "\n"
        return formatted_msg

    def log(self, level: int, msg: str, *args: object, stacklevel: int = 1) -> None:
        """
        Log a message at the specified level.

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
        _ = stacklevel
        match level:
            case _ if level >= logging.CRITICAL:
                self.critical(msg, args, stacklevel=stacklevel)
            case _ if level >= logging.ERROR:
                self.error(msg, args, stacklevel=stacklevel)
            case _ if level >= logging.WARNING:
                self.warning(msg, args, stacklevel=stacklevel)
            case _ if level >= logging.INFO:
                self.info(msg, args, stacklevel=stacklevel)
            case _:
                self.debug(msg, args, stacklevel=stacklevel)

    def debug(self, msg: str, *args: object, stacklevel: int = 1) -> None:
        """
        Log a debug-level message.

        Parameters
        ----------
        msg : str
            The debug message.
        *args : object
            Arguments to format the message.
        stacklevel : int, optional
            The stack level for the log (default is 1).
        """
        _ = stacklevel
        if not self._disable_console_output and self._logging_level <= logging.DEBUG:
            sys.stderr.write(self._format(msg, *args))

    def info(self, msg: str, *args: object, stacklevel: int = 1) -> None:
        """
        Log an info-level message.

        Parameters
        ----------
        msg : str
            The info message.
        *args : object
            Arguments to format the message.
        stacklevel : int, optional
            The stack level for the log (default is 1).
        """
        _ = stacklevel
        if not self._disable_console_output and self._logging_level <= logging.INFO:
            sys.stderr.write(self._format(msg, *args))

    def warning(self, msg: str, *args: object, stacklevel: int = 1) -> None:
        """
        Log a warning-level message.

        Parameters
        ----------
        msg : str
            The warning message.
        *args : object
            Arguments to format the message.
        stacklevel : int, optional
            The stack level for the log (default is 1).
        """
        _ = stacklevel
        if not self._disable_console_output and self._logging_level <= logging.WARNING:
            sys.stderr.write(self._format(msg, *args))

    def error(self, msg: str, *args: object, stacklevel: int = 1) -> None:
        """
        Log an error-level message.

        Parameters
        ----------
        msg : str
            The error message.
        *args : object
            Arguments to format the message.
        stacklevel : int, optional
            The stack level for the log (default is 1).
        """
        _ = stacklevel
        if not self._disable_console_output and self._logging_level <= logging.ERROR:
            sys.stderr.write(self._format(msg, *args))

    def critical(self, msg: str, *args: object, stacklevel: int = 1) -> None:
        """
        Log a critical-level message.

        Parameters
        ----------
        msg : str
            The critical message.
        *args : object
            Arguments to format the message.
        stacklevel : int, optional
            The stack level for the log (default is 1).
        """
        _ = stacklevel
        if not self._disable_console_output and self._logging_level <= logging.CRITICAL:
            sys.stderr.write(self._format(msg, *args))

    def suppress_console_output(self) -> None:
        """
        Disable console output.
        """
        self._disable_console_output = False

    def enable_console_output(self) -> None:
        """
        Enable console output.
        """
        self._disable_console_output = True

    def set_logging_level(self, logging_level: int | str) -> None:
        """
        Set the logging level.

        Parameters
        ----------
        logging_level : int or str
            The logging level to set, either as an integer or a string.
        """
        if isinstance(logging_level, int):
            self._logging_level = logging_level
        else:
            self._logging_level = self._logging_level_map[logging_level]

    def setLevel(self, level: int | str) -> None:  # noqa: N802
        """
        Alias for set_logging_level.

        Parameters
        ----------
        level : int or str
            The logging level to set, either as an integer or a string.
        """
        self.set_logging_level(level)

    @property
    def logging_level(self) -> int:
        """
        Get the current logging level.

        Returns
        -------
        int
            The current logging level.
        """
        return self._logging_level


def is_running_in_pytest() -> bool:
    """
    Check if the code is running inside pytest.

    Returns
    -------
    bool
        True if running in pytest, False otherwise.
    """
    return "pytest" in sys.modules


_event_logger: logging.Logger | FakeLogger = FakeLogger(disable_console_output=is_running_in_pytest())


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
        fmt = DEFAULT_LOG_FORMATTER
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
