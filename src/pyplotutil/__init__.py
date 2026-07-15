"""Plotting and data handling utility package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pyplotutil.datautil import BaseData, Data, Dataset, TaggedData
from pyplotutil.loggingutil import (
    FakeLogger,
    check_level,
    event_logger,
    evlog,
    get_event_logger_filename,
    get_logging_level_from_verbose_count,
    start_event_logging,
    start_logging,
)
from pyplotutil.plotutil import (
    annotate_with_arrow,
    apply_ieee_style,
    apply_nature_style,
    apply_notebook_style,
    apply_science_style,
    apply_style,
    calculate_mean_err,
    compatible_filename,
    extract_common_path,
    fill_between_err,
    get_limits,
    get_tlim_mask,
    make_figure_paths,
    plot_mean_err,
    plot_multi_timeseries,
    save_figure,
)

try:
    __version__ = version("pyplotutil")
except PackageNotFoundError:  # pragma: no cover -- only when the package is not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "BaseData",
    "Data",
    "Dataset",
    "FakeLogger",
    "TaggedData",
    "__version__",
    "annotate_with_arrow",
    "apply_ieee_style",
    "apply_nature_style",
    "apply_notebook_style",
    "apply_science_style",
    "apply_style",
    "calculate_mean_err",
    "check_level",
    "compatible_filename",
    "event_logger",
    "evlog",
    "extract_common_path",
    "fill_between_err",
    "get_event_logger_filename",
    "get_limits",
    "get_logging_level_from_verbose_count",
    "get_tlim_mask",
    "make_figure_paths",
    "plot_mean_err",
    "plot_multi_timeseries",
    "save_figure",
    "start_event_logging",
    "start_logging",
]
