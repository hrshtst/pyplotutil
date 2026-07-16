"""Plotting Utilities for Scientific Data Visualization.

This module provides a collection of utilities for creating and saving scientific plots,
particularly focused on time series data visualization with error representations.

Key Features
-----------
- Save figures with multiple file formats
- Plot multiple time series with customizable styles
- Error visualization (standard deviation, variance, range, standard error, confidence interval)
- Plot directly from Dataset and TaggedData objects
- Path handling utilities for figure organization

Examples
--------
Basic figure saving:
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [1, 2, 3])
>>> save_figure(fig, "output_directory", "my_plot", ".png")

Time series with error bars:
>>> t = np.linspace(0, 10, 100)
>>> data = np.random.randn(5, 100)  # 5 trials, 100 timepoints
>>> plot_mean_err(ax, t, data, err_type="std", tlim=(0, 5))

Multiple time series:
>>> y_arr = np.array([np.sin(t), np.cos(t)])
>>> plot_multi_timeseries(ax, t, y_arr, labels=["sin", "cos"])

Notes
-----
- All plotting functions return matplotlib objects for further customization
- Error calculations support various statistical measures
- File paths are sanitized for cross-platform compatibility

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from scipy import stats

from pyplotutil._typing import NoDefault, no_default
from pyplotutil.datautil import Dataset, TaggedData
from pyplotutil.loggingutil import evlog

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Annotation
    from matplotlib.typing import ColorType

    from pyplotutil._typing import FilePath, Unknown


FilePathT = TypeVar("FilePathT", str, Path)


def _style_options(
    *,
    grid: bool = False,
    scatter: bool = False,
    no_latex: bool = False,
    cjk_jp_font: bool = False,
) -> list[str]:
    """Generate a list of style options based on input parameters.

    Parameters
    ----------
    grid : bool, optional
        Enable grid style, by default False
    scatter : bool, optional
        Enable scatter style, by default False
    no_latex : bool, optional
        Disable LaTeX rendering, by default False
    cjk_jp_font : bool, optional
        Enable CJK Japanese font support, by default False

    Returns
    -------
    list[str]
        List of style options based on enabled parameters

    """
    styles: list[str] = []
    if grid:
        styles.append("grid")
    if scatter:
        styles.append("scatter")
    if no_latex:
        styles.append("no-latex")
    if cjk_jp_font:
        styles.append("cjk-jp-font")
    return styles


def apply_science_style(
    *,
    grid: bool = False,
    scatter: bool = False,
    no_latex: bool = False,
    cjk_jp_font: bool = False,
) -> None:
    """Apply science style to matplotlib plots.

    Parameters
    ----------
    grid : bool, optional
        Enable grid style, by default False
    scatter : bool, optional
        Enable scatter style, by default False
    no_latex : bool, optional
        Disable LaTeX rendering, by default False
    cjk_jp_font : bool, optional
        Enable CJK Japanese font support, by default False

    """
    styles = ["science"]
    styles.extend(_style_options(grid=grid, scatter=scatter, no_latex=no_latex, cjk_jp_font=cjk_jp_font))
    plt.style.use(styles)


def apply_ieee_style(
    *,
    grid: bool = False,
    scatter: bool = False,
    no_latex: bool = False,
    cjk_jp_font: bool = False,
) -> None:
    """Apply IEEE style to matplotlib plots.

    Parameters
    ----------
    grid : bool, optional
        Enable grid style, by default False
    scatter : bool, optional
        Enable scatter style, by default False
    no_latex : bool, optional
        Disable LaTeX rendering, by default False
    cjk_jp_font : bool, optional
        Enable CJK Japanese font support, by default False

    """
    styles = ["science", "ieee"]
    styles.extend(_style_options(grid=grid, scatter=scatter, no_latex=no_latex, cjk_jp_font=cjk_jp_font))
    plt.style.use(styles)
    plt.rcParams.update({"figure.dpi": "100"})


def apply_nature_style(
    *,
    grid: bool = False,
    scatter: bool = False,
    no_latex: bool = False,
    cjk_jp_font: bool = False,
) -> None:
    """Apply Nature journal style to matplotlib plots.

    Parameters
    ----------
    grid : bool, optional
        Enable grid style, by default False
    scatter : bool, optional
        Enable scatter style, by default False
    no_latex : bool, optional
        Disable LaTeX rendering, by default False
    cjk_jp_font : bool, optional
        Enable CJK Japanese font support, by default False

    """
    styles = ["science", "nature"]
    styles.extend(_style_options(grid=grid, scatter=scatter, no_latex=no_latex, cjk_jp_font=cjk_jp_font))
    plt.style.use(styles)


def apply_notebook_style(
    *,
    grid: bool = False,
    scatter: bool = False,
    no_latex: bool = False,
    cjk_jp_font: bool = False,
) -> None:
    """Apply Jupyter notebook style to matplotlib plots.

    Parameters
    ----------
    grid : bool, optional
        Enable grid style, by default False
    scatter : bool, optional
        Enable scatter style, by default False
    no_latex : bool, optional
        Disable LaTeX rendering, by default False
    cjk_jp_font : bool, optional
        Enable CJK Japanese font support, by default False

    """
    styles = ["science", "notebook"]
    styles.extend(_style_options(grid=grid, scatter=scatter, no_latex=no_latex, cjk_jp_font=cjk_jp_font))
    plt.style.use(styles)


def apply_style(
    style: Literal["science", "ieee", "nature", "notebook"],
    *,
    grid: bool = False,
    scatter: bool = False,
    no_latex: bool = False,
    cjk_jp_font: bool = False,
) -> None:
    """Apply specific style to matplotlib plots.

    Parameters
    ----------
    style: Literal["science", "ieee", "nature", "notebook"]
        Style name to apply. Choose from "science", "ieee", "nature", "notebook".
    grid : bool, optional
        Enable grid style, by default False
    scatter : bool, optional
        Enable scatter style, by default False
    no_latex : bool, optional
        Disable LaTeX rendering, by default False
    cjk_jp_font : bool, optional
        Enable CJK Japanese font support, by default False

    """
    if style == "science":
        apply_science_style(grid=grid, scatter=scatter, no_latex=no_latex, cjk_jp_font=cjk_jp_font)
    elif style == "ieee":
        apply_ieee_style(grid=grid, scatter=scatter, no_latex=no_latex, cjk_jp_font=cjk_jp_font)
    elif style == "nature":
        apply_nature_style(grid=grid, scatter=scatter, no_latex=no_latex, cjk_jp_font=cjk_jp_font)
    elif style == "notebook":
        apply_notebook_style(grid=grid, scatter=scatter, no_latex=no_latex, cjk_jp_font=cjk_jp_font)
    else:
        msg = f"Unsupported style: {style}"
        evlog().error(msg)
        raise ValueError(msg)


def compatible_filename(filename: FilePathT) -> FilePathT:
    """Convert filename to a compatible format by replacing special characters.

    Parameters
    ----------
    filename : str or Path
        The input filename to be converted.

    Returns
    -------
    str or Path
        The converted filename with special characters replaced.

    """
    table = {":": "", " ": "_", "(": "", ")": "", "+": "x", "=": "-"}
    compat_filename = str(filename).translate(str.maketrans(table))  # type: ignore[arg-type]
    if compat_filename != str(filename):
        evlog().debug("Filename has been converted to compatible one.")
        evlog().debug("      given filename: %s", filename)
        evlog().debug("  converted filename: %s", compat_filename)
    return type(filename)(compat_filename)


def make_figure_paths(
    output_directory: FilePath,
    basename: str,
    extensions: str | Iterable[str],
    *,
    separate_dir_by_main_module: bool | str,
    separate_dir_by_ext: bool,
) -> list[Path]:
    """Generate figure file paths based on given parameters.

    Parameters
    ----------
    output_directory : str or Path
        Directory where figures will be saved.
    basename : str
        Base name for the figure files.
    extensions : str or Iterable[str]
        File extensions to use.
    separate_dir_by_main_module : bool or str
        Whether to create separate directory by main module name.
    separate_dir_by_ext : bool
        Whether to separate files by extension in different directories.

    Returns
    -------
    list[Path]
        List of generated figure paths.

    """
    if isinstance(extensions, str):
        extensions = [extensions]
    # Make a generator that ensures each extension starts with '.', and remove duplicates.
    extensions = {x if x.startswith(".") else f".{x}" for x in extensions}

    main_module_name = None
    if isinstance(separate_dir_by_main_module, str):
        main_module_name = separate_dir_by_main_module
    elif separate_dir_by_main_module:
        try:
            import __main__  # noqa: PLC0415  # must resolve the running script at call time

            main_module_name = Path(__main__.__file__).stem
        except ImportError:
            main_module_name = None

    built_path = Path(output_directory)
    if main_module_name:
        built_path /= main_module_name
    built_path /= basename
    if built_path.suffix.startswith(".") and built_path.suffix[1].isdigit():
        # When data_file_path.suffix starts with a digit it's not a suffix.
        # >>> Path('awesome_ratio-2.5').with_suffix('.svg')
        # 'awesome_ratio-2.svg'  # this is wrong filename
        built_path = built_path.with_name(built_path.name + ".x")

    figure_paths: list[Path] = []
    for ext in extensions:
        figure_path = built_path
        if separate_dir_by_ext:
            figure_path = built_path.parent / ext[1:] / built_path.name
        figure_paths.append(compatible_filename(figure_path.with_suffix(ext)))
    return figure_paths


def save_figure(
    fig: Figure,
    output_directory: FilePath,
    basename: str,
    extensions: str | Iterable[str] | None,
    *,
    separate_dir_by_main_module: bool | str = False,
    separate_dir_by_ext: bool = False,
    make_directories: bool = True,
    dpi: float | Literal["figure"] = "figure",
    bbox_inches: Literal["tight"] | None = "tight",
    pad_inches: float | Literal["layout"] = 0.1,
) -> list[Path]:
    """Save figures to specified paths with given parameters.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure object to save.
    output_directory : str or Path
        Directory where figures will be saved.
    basename : str
        Base name for the figure files.
    extensions : str or Iterable[str] or None
        File extensions to use.
    separate_dir_by_main_module : bool or str, optional
        Whether to create separate directory by main module name, by default False.
    separate_dir_by_ext : bool, optional
        Whether to separate files by extension, by default False.
    make_directories : bool, optional
        Whether to create directories if they don't exist, by default True.
    dpi : float or "figure", optional
        Resolution of the output figure, by default "figure".
    bbox_inches : "tight" or None, optional
        Bounding box in inches, by default "tight".
    pad_inches : float or "layout", optional
        Padding in inches, by default 0.1.

    Returns
    -------
    list[Path]
        List of paths where figures were saved.

    Raises
    ------
    ValueError
        If output_directory is None.

    """
    if output_directory is None:
        msg = f"'None' is not allowed for directory path: {output_directory}, {type(output_directory)}"
        evlog().critical(msg)
        raise ValueError(msg)

    if extensions is None:
        evlog().warning("Nothing saved.")
        evlog().debug("Figures have not been saved since no extension is provided.")
        return []

    figure_paths = make_figure_paths(
        output_directory,
        basename,
        extensions,
        separate_dir_by_main_module=separate_dir_by_main_module,
        separate_dir_by_ext=separate_dir_by_ext,
    )
    if make_directories:
        for directory_path in {p.parent for p in figure_paths}:
            directory_path.mkdir(parents=True, exist_ok=True)
            evlog().debug("Directory created: %s", str(directory_path))

    for figure_path in figure_paths:
        fig.savefig(figure_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        evlog().info("Figure saved: %s", str(figure_path))

    return figure_paths


def extract_common_path(*paths: str | Path) -> Path:
    """Extract the common path from multiple file paths.

    Parameters
    ----------
    *paths : str or Path
        Variable number of path arguments.

    Returns
    -------
    Path
        Common path shared between all input paths.

    """
    are_absolute = [x.is_absolute() for x in map(Path, paths)]
    if not all(are_absolute) and any(are_absolute):
        # When absolute and relative paths are mixed, convert them to absolute ones.
        path_objects = [Path(path).resolve() for path in paths]
    else:
        # Convert paths to Path objects
        path_objects = [Path(path) for path in paths]

    # Find the shortest path
    shortest_path = min(path_objects, key=lambda p: len(p.parts))

    # Iterate over the shortest path's parts
    common_parts = []
    for i, part in enumerate(shortest_path.parts):
        if all(part in path.parts[: i + 1] for path in path_objects):
            common_parts.append(part)
        else:
            break

    # Join common parts back into a path object
    common_path = Path(*common_parts)
    if common_path.is_file():
        # When common path exists and it is a file, its parent directory is returned.
        common_path = common_path.parent
    return common_path


def _get_limits(
    xlim: Sequence[float] | None,
    fallback: tuple[float, float] | None,
    fallback_xlim: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Calculate axis limits based on input sequence and fallback values.

    Parameters
    ----------
    xlim : Sequence[float] | None
        Input sequence of values to determine limits from
    fallback : tuple[float, float] | None
        Default fallback limits to use if xlim is empty
    fallback_xlim : tuple[float, float] | None
        Secondary fallback limits that override primary fallback

    Returns
    -------
    tuple[float, float] | None
        Calculated axis limits as (min, max) tuple, or None if xlim is None

    """
    if xlim is None:
        return None

    fixed_xlim: tuple[float, float] | None = None
    if len(xlim) == 0:
        if fallback is not None:
            fixed_xlim = fallback
        if fallback_xlim is not None:
            fixed_xlim = fallback_xlim
    elif len(xlim) == 1:
        fixed_xlim = (-abs(xlim[0]), abs(xlim[0]))
    else:
        fixed_xlim = (min(xlim), max(xlim))
    return fixed_xlim


@overload
def get_limits(
    xlim: Sequence[float] | None,
    ylim: Sequence[float] | None,
    *,
    fallback: tuple[float, float] | None = None,
    fallback_xlim: tuple[float, float] | None = None,
    fallback_ylim: tuple[float, float] | None = None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]: ...


@overload
def get_limits(
    xlim: Sequence[float] | None,
    ylim: NoDefault = no_default,
    *,
    fallback: tuple[float, float] | None = None,
    fallback_xlim: tuple[float, float] | None = None,
    fallback_ylim: tuple[float, float] | None = None,
) -> tuple[float, float] | None: ...


def get_limits(
    xlim: Sequence[float] | None,
    ylim: Sequence[float] | None | NoDefault = no_default,
    *,
    fallback: tuple[float, float] | None = None,
    fallback_xlim: tuple[float, float] | None = None,
    fallback_ylim: tuple[float, float] | None = None,
) -> tuple[float, float] | None | tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Calculate axis limits for one or two dimensions.

    Parameters
    ----------
    xlim : Sequence[float] | None
        Input sequence for x-axis limits
    ylim : Sequence[float] | None | NoDefault, optional
        Input sequence for y-axis limits
    fallback : tuple[float, float] | None, optional
        Default fallback limits for both axes
    fallback_xlim : tuple[float, float] | None, optional
        Specific fallback limits for x-axis
    fallback_ylim : tuple[float, float] | None, optional
        Specific fallback limits for y-axis

    Returns
    -------
    tuple[float, float] | None | tuple[tuple[float, float] | None, tuple[float, float] | None]
        Single axis limits or tuple of (x_limits, y_limits)

    """
    fixed_xlim = _get_limits(xlim, fallback, fallback_xlim)
    if ylim is no_default:
        return fixed_xlim
    fixed_ylim = _get_limits(ylim, fallback, fallback_ylim)
    return fixed_xlim, fixed_ylim


def get_tlim_mask(t: np.ndarray, tlim: tuple[float, float] | None) -> np.ndarray:
    """Create a boolean mask for time limits.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    tlim : tuple[float, float] or None
        Time limits (min, max).

    Returns
    -------
    np.ndarray
        Boolean mask array.

    """
    return np.full(t.shape, fill_value=True) if tlim is None else (t >= tlim[0]) & (t <= tlim[1])


def calculate_mean_err(
    data_array: np.ndarray,
    err_type: str = "std",
    ddof: int = 0,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Calculate mean and error metrics for data array.

    Parameters
    ----------
    data_array : np.ndarray
        Input data array.
    err_type : str, optional
        Type of error to calculate ("std", "var", "range", "se", "ci"), by default "std".
    ddof : int, optional
        Delta degrees of freedom, by default 0. For confidence intervals the sample standard
        deviation (ddof=1) is conventional.
    confidence : float, optional
        Confidence level used when err_type is "ci", by default 0.95.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
        Mean, error1, and error2 (if applicable) arrays.

    Raises
    ------
    TypeError
        If err_type is not a string.
    ValueError
        If err_type is not recognized, if confidence is not between 0 and 1, or if a confidence
        interval is requested with fewer than two trials.

    """
    if not isinstance(err_type, str):
        msg = f"`err_type` must be string: {err_type}, (type: {type(err_type)})"
        raise TypeError(msg)

    mean = np.mean(data_array, axis=0)
    if err_type.lower() in ("std", "sd"):
        # standard deviation
        std = np.std(data_array, axis=0, ddof=ddof)
        return mean, std, None

    if err_type.lower() == "var":
        # variance
        var = np.var(data_array, axis=0, ddof=ddof)
        return mean, var, None

    if err_type.lower() == "range":
        # range
        lower = mean - np.min(data_array, axis=0)
        upper = np.max(data_array, axis=0) - mean
        return mean, lower, upper

    if err_type.lower() == "se":
        # standard error of the mean: deviation over the number of trials
        std = np.std(data_array, axis=0, ddof=ddof)
        se = std / np.sqrt(data_array.shape[0])
        return mean, se, None

    if err_type.lower() == "ci":
        # confidence interval of the mean, scaled by a Student's t critical value
        if not 0.0 < confidence < 1.0:
            msg = f"`confidence` must be between 0 and 1 exclusive: {confidence}"
            raise ValueError(msg)
        n_trials = data_array.shape[0]
        min_trials = 2
        if n_trials < min_trials:
            msg = f"confidence interval requires at least {min_trials} trials: {n_trials}"
            raise ValueError(msg)
        std = np.std(data_array, axis=0, ddof=ddof)
        se = std / np.sqrt(n_trials)
        t_crit = float(stats.t.ppf(0.5 + 0.5 * confidence, df=n_trials - 1))
        return mean, t_crit * se, None

    msg = f"unrecognized error type: {err_type}"
    raise ValueError(msg)


def _normalize_labels(labels: str | Iterable[str] | None, n_lines: int) -> list[str]:
    """Return one label per line, expanding a single string with an index suffix.

    Parameters
    ----------
    labels : str or Iterable[str] or None
        Labels for lines. None yields index labels; a single string is expanded per line.
    n_lines : int
        Number of lines to label.

    Returns
    -------
    list[str]
        One label per line.

    """
    if labels is None:
        return [f"{i}" for i in range(n_lines)]
    if isinstance(labels, str):
        return [labels] if n_lines == 1 else [f"{labels}_{i}" for i in range(n_lines)]
    return [str(label) for label in labels]


def _dataset_labels(dataset: Dataset) -> list[str] | None:
    """Return file stem labels for a dataset, or None when it was not loaded from files.

    Parameters
    ----------
    dataset : Dataset
        Dataset to derive line labels from.

    Returns
    -------
    list[str] or None
        File stems of the data files, or None when unavailable.

    """
    try:
        return [path.stem for path in dataset.datapaths]
    except AttributeError:
        return None


def _plot_tagged_timeseries(
    ax: Axes,
    tagged: TaggedData,
    column: str,
    *,
    tlim: tuple[float, float] | None,
    lw: int | None,
    color: ColorType | None,
    fmt: str | None,
    cmap_name: str | None,
    t_axis_name: str,
    t_shift: float,
) -> list[Line2D]:
    """Plot one line per tag group of a TaggedData object.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    tagged : TaggedData
        Tagged data whose groups are plotted as separate lines.
    column : str
        Name of the column to plot.
    tlim : tuple[float, float] or None
        Time limits.
    lw : int or None
        Line width.
    color : ColorType or None
        Line color shared by all lines.
    fmt : str or None
        Format string.
    cmap_name : str or None
        Colormap name used to assign one color per tag.
    t_axis_name : str
        Name of the time axis column in each group.
    t_shift : float
        Time shift subtracted from the time values.

    Returns
    -------
    list[Line2D]
        List of plotted lines, one per tag in alphabetical order.

    """
    cmap = plt.get_cmap(cmap_name) if cmap_name is not None else None
    lines: list[Line2D] = []
    for i, (tag, data) in enumerate(sorted(tagged.items(), key=lambda item: item[0])):
        line_color = color
        if line_color is None and cmap is not None:
            line_color = cmap(i)
        t_arr = data[t_axis_name].to_numpy() - t_shift
        lines.extend(
            plot_multi_timeseries(
                ax,
                t_arr,
                data[column].to_numpy(),
                tlim=tlim,
                lw=lw,
                color=line_color,
                fmt=fmt,
                labels=str(tag),
            ),
        )
    return lines


@overload
def plot_multi_timeseries(
    ax: Axes,
    t: np.ndarray,
    y_arr: np.ndarray,
    *,
    tlim: tuple[float, float] | None = None,
    lw: int | None = None,
    color: ColorType | None = None,
    fmt: str | None = None,
    labels: str | Iterable[str] | None = None,
    cmap_name: str | None = None,
    t_axis_name: str = "t",
    t_shift: float = 0.0,
) -> list[Line2D]: ...


@overload
def plot_multi_timeseries(
    ax: Axes,
    t: Dataset | TaggedData,
    y_arr: str,
    *,
    tlim: tuple[float, float] | None = None,
    lw: int | None = None,
    color: ColorType | None = None,
    fmt: str | None = None,
    labels: str | Iterable[str] | None = None,
    cmap_name: str | None = None,
    t_axis_name: str = "t",
    t_shift: float = 0.0,
) -> list[Line2D]: ...


def plot_multi_timeseries(
    ax,
    t,
    y_arr,
    *,
    tlim=None,
    lw=None,
    color=None,
    fmt=None,
    labels=None,
    cmap_name=None,
    t_axis_name="t",
    t_shift=0.0,
):
    """Plot multiple time series on the same axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    t : np.ndarray or Dataset or TaggedData
        Array of time values, or a data object to plot from directly. A Dataset plots one line
        per data file; a TaggedData plots one line per tag, labeled by the tag.
    y_arr : np.ndarray or str
        Array of y values, or the column name when plotting from a data object.
    tlim : tuple[float, float] or None, optional
        Time limits, by default None.
    lw : int or None, optional
        Line width, by default None.
    color : ColorType or None, optional
        Line color, by default None.
    fmt : str or None, optional
        Format string, by default None.
    labels : str or Iterable[str] or None, optional
        Labels for lines, by default None. When plotting from a Dataset, file stems are used
        as fallback labels.
    cmap_name : str or None, optional
        Colormap name, by default None.
    t_axis_name : str, optional
        Name of the time axis column, by default "t". Only used with data objects.
    t_shift : float, optional
        Time shift subtracted from the time values, by default 0.0. Only used with data objects.

    Returns
    -------
    list[Line2D]
        List of plotted lines.

    """
    if isinstance(t, TaggedData):
        return _plot_tagged_timeseries(
            ax,
            t,
            y_arr,
            tlim=tlim,
            lw=lw,
            color=color,
            fmt=fmt,
            cmap_name=cmap_name,
            t_axis_name=t_axis_name,
            t_shift=t_shift,
        )
    if isinstance(t, Dataset):
        if labels is None:
            labels = _dataset_labels(t)
        t, y_arr = t.get_timeseries(y_arr, t_shift=t_shift, t_axis_name=t_axis_name)

    mask = get_tlim_mask(t, tlim)
    y_arr = np.atleast_2d(y_arr)
    cmap = plt.get_cmap(cmap_name) if cmap_name is not None else None
    labels = _normalize_labels(labels, len(y_arr))

    kwargs: dict[str, Unknown] = {}
    if lw is not None:
        kwargs["lw"] = lw
    if color is not None:
        kwargs["c"] = color

    t_mask = t[mask]
    lines: list[Line2D] = []
    for i, (y, label) in enumerate(zip(y_arr, labels, strict=True)):
        kwargs["label"] = label
        if color is None and cmap is not None:
            kwargs["c"] = cmap(i)

        if fmt is None:  # noqa: SIM108
            _lines = ax.plot(t_mask, y[mask], **kwargs)
        else:
            _lines = ax.plot(t_mask, y[mask], fmt, **kwargs)
        lines.extend(_lines)
    return lines


@overload
def plot_mean_err(
    ax: Axes,
    t: np.ndarray,
    y_arr: np.ndarray,
    err_type: str | None = "std",
    *,
    tlim: tuple[float, float] | None = None,
    lw: int | None = None,
    capsize: int | None = None,
    color: ColorType | None = None,
    fmt: str | None = None,
    label: str | None = None,
    ddof: int = 0,
    confidence: float = 0.95,
    t_axis_name: str = "t",
    t_shift: float = 0.0,
) -> Line2D: ...


@overload
def plot_mean_err(
    ax: Axes,
    t: Dataset,
    y_arr: str,
    err_type: str | None = "std",
    *,
    tlim: tuple[float, float] | None = None,
    lw: int | None = None,
    capsize: int | None = None,
    color: ColorType | None = None,
    fmt: str | None = None,
    label: str | None = None,
    ddof: int = 0,
    confidence: float = 0.95,
    t_axis_name: str = "t",
    t_shift: float = 0.0,
) -> Line2D: ...


def plot_mean_err(
    ax,
    t,
    y_arr,
    err_type="std",
    *,
    tlim=None,
    lw=None,
    capsize=None,
    color=None,
    fmt=None,
    label=None,
    ddof=0,
    confidence=0.95,
    t_axis_name="t",
    t_shift=0.0,
):
    """Plot mean with error bars.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    t : np.ndarray or Dataset
        Array of time values, or a Dataset whose column is averaged across data files.
    y_arr : np.ndarray or str
        Array of y values, or the column name when plotting from a Dataset.
    err_type : str or None, optional
        Type of error to plot, by default "std".
    tlim : tuple[float, float] or None, optional
        Time limits, by default None.
    lw : int or None, optional
        Line width, by default None.
    capsize : int or None, optional
        Size of error bar caps, by default None.
    color : ColorType or None, optional
        Line color, by default None.
    fmt : str or None, optional
        Format string, by default None.
    label : str or None, optional
        Label for the plot, by default None.
    ddof : int, optional
        Delta degrees of freedom forwarded to the error calculation, by default 0.
    confidence : float, optional
        Confidence level used when err_type is "ci", by default 0.95.
    t_axis_name : str, optional
        Name of the time axis column, by default "t". Only used with a Dataset.
    t_shift : float, optional
        Time shift subtracted from the time values, by default 0.0. Only used with a Dataset.

    Returns
    -------
    Line2D
        The plotted line.

    """
    if isinstance(t, Dataset):
        t, y_arr = t.get_timeseries(y_arr, t_shift=t_shift, t_axis_name=t_axis_name)

    mask = get_tlim_mask(t, tlim)
    y_arr = np.atleast_2d(y_arr)

    kwargs: dict[str, Unknown] = {}
    if lw is not None:
        kwargs["lw"] = lw
    if capsize is not None:
        kwargs["capsize"] = capsize
    if color is not None:
        kwargs["c"] = color
    if fmt is not None:
        kwargs["fmt"] = fmt
    kwargs["label"] = label

    if err_type is None or err_type == "none":
        mean, _, _ = calculate_mean_err(y_arr)
        lines = plot_multi_timeseries(ax, t, mean, tlim=tlim, lw=lw, color=color, fmt=fmt, labels=label)
    else:
        mean, err1, err2 = calculate_mean_err(y_arr, err_type=err_type, ddof=ddof, confidence=confidence)
        if err2 is None:
            eb = ax.errorbar(t[mask], mean[mask], yerr=err1[mask], **kwargs)
        else:
            eb = ax.errorbar(t[mask], mean[mask], yerr=(err1[mask], err2[mask]), **kwargs)
        lines = [eb.lines[0]]
    return lines[0]


@overload
def fill_between_err(
    ax: Axes,
    t: np.ndarray,
    y_arr: np.ndarray,
    err_type: str | None = "std",
    *,
    tlim: tuple[float, float] | None = None,
    color: ColorType | None = None,
    alpha: float | None = None,
    interpolate: bool = False,
    suppress_exception: bool = False,
    ddof: int = 0,
    confidence: float = 0.95,
    t_axis_name: str = "t",
    t_shift: float = 0.0,
) -> Axes: ...


@overload
def fill_between_err(
    ax: Axes,
    t: Dataset,
    y_arr: str,
    err_type: str | None = "std",
    *,
    tlim: tuple[float, float] | None = None,
    color: ColorType | None = None,
    alpha: float | None = None,
    interpolate: bool = False,
    suppress_exception: bool = False,
    ddof: int = 0,
    confidence: float = 0.95,
    t_axis_name: str = "t",
    t_shift: float = 0.0,
) -> Axes: ...


def fill_between_err(
    ax,
    t,
    y_arr,
    err_type="std",
    *,
    tlim=None,
    color=None,
    alpha=None,
    interpolate=False,
    suppress_exception=False,
    ddof=0,
    confidence=0.95,
    t_axis_name="t",
    t_shift=0.0,
):
    """Fill between error bounds.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    t : np.ndarray or Dataset
        Array of time values, or a Dataset whose column is averaged across data files.
    y_arr : np.ndarray or str
        Array of y values, or the column name when plotting from a Dataset.
    err_type : str or None, optional
        Type of error to fill, by default "std".
    tlim : tuple[float, float] or None, optional
        Time limits, by default None.
    color : ColorType or None, optional
        Fill color, by default None.
    alpha : float or None, optional
        Fill transparency, by default None.
    interpolate : bool, optional
        Whether to use interpolate, by default False.
    suppress_exception : bool, optional
        Whether to suppress exceptions, by default False.
    ddof : int, optional
        Delta degrees of freedom forwarded to the error calculation, by default 0.
    confidence : float, optional
        Confidence level used when err_type is "ci", by default 0.95.
    t_axis_name : str, optional
        Name of the time axis column, by default "t". Only used with a Dataset.
    t_shift : float, optional
        Time shift subtracted from the time values, by default 0.0. Only used with a Dataset.

    Returns
    -------
    Axes
        The modified axes object.

    Raises
    ------
    ValueError
        If err_type is None and suppress_exception is False.

    """
    if err_type is None or err_type == "none":
        if suppress_exception:
            return ax
        msg = "`err_type` for `fill_between_err` must not be None."
        raise ValueError(msg)

    if isinstance(t, Dataset):
        t, y_arr = t.get_timeseries(y_arr, t_shift=t_shift, t_axis_name=t_axis_name)

    mask = get_tlim_mask(t, tlim)
    y_arr = np.atleast_2d(y_arr)

    kwargs: dict[str, Unknown] = {}
    if color is not None:
        kwargs["facecolor"] = color
    if alpha is not None:
        kwargs["alpha"] = alpha
    kwargs["interpolate"] = interpolate

    mean, err1, err2 = calculate_mean_err(y_arr, err_type=err_type, ddof=ddof, confidence=confidence)
    # Note that fill_between always goes behind lines.
    if err2 is None:
        ax.fill_between(t[mask], mean[mask] + err1[mask], mean[mask] - err1[mask], **kwargs)
    else:
        # err1 is the distance below the mean and err2 the distance above, as in `plot_mean_err`.
        ax.fill_between(t[mask], mean[mask] - err1[mask], mean[mask] + err2[mask], **kwargs)
    return ax


def mask_to_spans(t: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Return (start, end) pairs of contiguous True runs in a boolean mask.

    Parameters
    ----------
    t : np.ndarray
        One-dimensional array of time values.
    mask : np.ndarray
        Boolean array of the same shape as `t`.

    Returns
    -------
    list[tuple[float, float]]
        Start and end time of each contiguous True run. A run of length one yields a span
        whose start and end coincide.

    Raises
    ------
    ValueError
        If `t` is not one-dimensional or the shapes of `t` and `mask` differ.

    """
    t = np.asarray(t)
    mask_arr = np.asarray(mask, dtype=bool)
    if t.ndim != 1:
        msg = f"`t` must be one-dimensional: {t.ndim} dimensions given"
        raise ValueError(msg)
    if t.shape != mask_arr.shape:
        msg = f"`t` and `mask` must have the same shape: {t.shape}, {mask_arr.shape}"
        raise ValueError(msg)
    padded = np.concatenate(([False], mask_arr, [False]))
    edges = np.where(padded[:-1] != padded[1:])[0]
    starts = edges[0::2]
    ends = edges[1::2] - 1
    return [(float(t[i]), float(t[j])) for i, j in zip(starts, ends, strict=True)]


def shade_spans(
    ax: Axes,
    t: np.ndarray,
    mask: np.ndarray,
    *,
    color: ColorType | None = None,
    alpha: float | None = 0.2,
    **kwargs: Unknown,
) -> Axes:
    """Shade vertical spans wherever a boolean mask is True.

    Draws one `axvspan` per contiguous True run in `mask`, e.g. to highlight phases,
    events, or regimes in a time series plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    t : np.ndarray
        One-dimensional array of time values.
    mask : np.ndarray
        Boolean array of the same shape as `t` selecting the time points to shade.
    color : ColorType or None, optional
        Fill color of the spans, by default None (matplotlib default).
    alpha : float or None, optional
        Fill transparency, by default 0.2.
    **kwargs : Unknown
        Additional keyword arguments passed to `Axes.axvspan`.

    Returns
    -------
    Axes
        The modified axes object.

    """
    if color is not None:
        kwargs["color"] = color
    if alpha is not None:
        kwargs["alpha"] = alpha
    for start, end in mask_to_spans(t, mask):
        ax.axvspan(start, end, **kwargs)
    return ax


def _set_grid(ax: Axes, *, grid: bool | Literal["both", "x", "y"]) -> None:
    """Enable or disable the grid on the given axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    grid : bool or {"both", "x", "y"}
        True or an axis name enables the grid on that axis; False disables it.

    """
    if grid:
        axis = grid if isinstance(grid, str) else "both"
        ax.grid(visible=True, axis=axis)
    else:
        ax.grid(visible=False)


def setup_axes(
    ax: Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: bool | Literal["both", "x", "y"] | None = None,
    legend: bool | str = False,
    legend_ncols: int = 1,
    legend_framealpha: float = 0.8,
) -> Axes:
    """Configure axes cosmetics in one call.

    Every option defaults to leaving the corresponding setting untouched, so the function
    only applies what is passed.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    xlabel : str or None, optional
        Label of the x axis, by default None.
    ylabel : str or None, optional
        Label of the y axis, by default None.
    title : str or None, optional
        Title of the axes, by default None.
    xlim : tuple[float, float] or None, optional
        Limits of the x axis, by default None.
    ylim : tuple[float, float] or None, optional
        Limits of the y axis, by default None.
    grid : bool or {"both", "x", "y"} or None, optional
        True or an axis name enables the grid on that axis, False disables it, and None
        (the default) leaves it untouched.
    legend : bool or str, optional
        True shows a legend at the best location, a string shows it at that location,
        by default False.
    legend_ncols : int, optional
        Number of legend columns, by default 1.
    legend_framealpha : float, optional
        Transparency of the legend frame, by default 0.8.

    Returns
    -------
    Axes
        The configured axes object.

    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid is not None:
        _set_grid(ax, grid=grid)
    if legend:
        legend_kwargs: dict[str, Unknown] = {
            "loc": legend if isinstance(legend, str) else "best",
            "ncols": legend_ncols,
            "framealpha": legend_framealpha,
        }
        ax.legend(**legend_kwargs)
    return ax


def label_with_unit(
    label: str | None,
    unit: str | None = None,
    *,
    units: Mapping[str, str] | None = None,
) -> str:
    """Return an axis label with its unit appended in brackets, e.g. "Position [m]".

    Parameters
    ----------
    label : str or None
        The label text. None yields an empty string.
    unit : str or None, optional
        The unit text, by default None. Brackets are added unless already present.
    units : Mapping[str, str] or None, optional
        Mapping from lowercase label names to units, used to look up the unit when
        `unit` is not given, by default None.

    Returns
    -------
    str
        The label with its unit, or the bare label when no unit is known.

    """
    if label is None:
        return ""
    if unit is None and units is not None:
        unit = units.get(label.lower())
    if unit is None:
        return label
    if not (unit.startswith("[") and unit.endswith("]")):
        unit = f"[{unit}]"
    return f"{label} {unit}"


def _orientation(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> int:
    """Return the orientation of the ordered point triplet (p, q, r).

    Parameters
    ----------
    p : tuple[float, float]
        First point.
    q : tuple[float, float]
        Second point.
    r : tuple[float, float]
        Third point.

    Returns
    -------
    int
        0 for collinear, 1 for clockwise, and 2 for counterclockwise ordering.

    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val > 0:
        return 1
    if val < 0:
        return 2
    return 0


def _on_segment(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
    """Check whether point q lies on segment (p, r), assuming the points are collinear.

    Parameters
    ----------
    p : tuple[float, float]
        Segment start.
    q : tuple[float, float]
        Point to check.
    r : tuple[float, float]
        Segment end.

    Returns
    -------
    bool
        True if q lies on the segment.

    """
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])


def _segments_intersect(
    p1: tuple[float, float],
    q1: tuple[float, float],
    p2: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    """Check whether segments (p1, q1) and (p2, q2) intersect, including touching cases.

    Parameters
    ----------
    p1 : tuple[float, float]
        Start of the first segment.
    q1 : tuple[float, float]
        End of the first segment.
    p2 : tuple[float, float]
        Start of the second segment.
    q2 : tuple[float, float]
        End of the second segment.

    Returns
    -------
    bool
        True if the segments intersect.

    """
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    return o4 == 0 and _on_segment(p2, q1, q2)


def _make_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: ColorType,
    size: float,
    arrowstyle: str,
) -> Annotation:
    """Draw a head-only arrow from start to end in data coordinates.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    start : tuple[float, float]
        Arrow tail in data coordinates.
    end : tuple[float, float]
        Arrow head in data coordinates.
    color : ColorType
        Arrow color.
    size : float
        Arrowhead size (mutation scale).
    arrowstyle : str
        Arrow style name.

    Returns
    -------
    Annotation
        The created annotation.

    """
    return ax.annotate(
        "",
        xytext=start,
        xy=end,
        arrowprops={"arrowstyle": arrowstyle, "color": color, "lw": 0, "mutation_scale": size},
    )


def add_direction_arrows(
    ax: Axes,
    line: Line2D,
    *,
    positions: Iterable[float] | None = None,
    crossing: tuple[tuple[float, float], tuple[float, float]] | None = None,
    size: float = 6.0,
    color: ColorType | None = None,
    arrowstyle: str = "-|>",
    reverse: bool = False,
    which: Literal["first", "all"] = "all",
    min_spacing: float = 1e-3,
    seen_crossings: list[tuple[float, float]] | None = None,
) -> list[Annotation]:
    """Add arrowheads to a plotted line indicating its direction of travel.

    Arrows are placed either at fractional arc-length positions along the line, or where
    the line crosses an auxiliary segment (useful for phase portraits). When neither
    `positions` nor `crossing` is given, a single arrow is placed halfway along the line.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    line : Line2D
        The plotted line to decorate.
    positions : Iterable[float] or None, optional
        Fractional positions along the line between 0 and 1, by default None.
    crossing : tuple[tuple[float, float], tuple[float, float]] or None, optional
        End points of an auxiliary segment; an arrow is drawn on every line segment that
        crosses it, by default None. Mutually exclusive with `positions`.
    size : float, optional
        Arrowhead size (mutation scale), by default 6.0.
    color : ColorType or None, optional
        Arrow color, by default None, which follows the line color.
    arrowstyle : str, optional
        Arrow style name, by default "-|>".
    reverse : bool, optional
        Point the arrows against the data order, by default False.
    which : {"first", "all"}, optional
        Whether to draw an arrow at only the first crossing or at all crossings,
        by default "all". Only used with `crossing`.
    min_spacing : float, optional
        Crossings closer than this to an already processed crossing are skipped,
        by default 1e-3. Only used with `crossing`.
    seen_crossings : list[tuple[float, float]] or None, optional
        Mutable record of processed crossings, shared between calls to avoid overlapping
        arrows from multiple lines, by default None. Only used with `crossing`.

    Returns
    -------
    list[Annotation]
        The created arrow annotations.

    Raises
    ------
    ValueError
        If both `positions` and `crossing` are given, if a position is outside [0, 1],
        or if the line has fewer than two points.

    """
    if positions is not None and crossing is not None:
        msg = "`positions` and `crossing` are mutually exclusive."
        raise ValueError(msg)
    if color is None:
        color = line.get_color()
    xdata = np.ravel(line.get_xdata())
    ydata = np.ravel(line.get_ydata())
    min_points = 2
    if len(xdata) < min_points:
        msg = f"`line` must have at least {min_points} points: {len(xdata)}"
        raise ValueError(msg)
    if crossing is not None:
        return _arrows_at_crossings(
            ax,
            xdata,
            ydata,
            crossing,
            color=color,
            size=size,
            arrowstyle=arrowstyle,
            reverse=reverse,
            which=which,
            min_spacing=min_spacing,
            seen_crossings=seen_crossings,
        )
    if positions is None:
        positions = (0.5,)
    return _arrows_at_positions(
        ax,
        xdata,
        ydata,
        positions,
        color=color,
        size=size,
        arrowstyle=arrowstyle,
        reverse=reverse,
    )


def _arrows_at_positions(
    ax: Axes,
    xdata: np.ndarray,
    ydata: np.ndarray,
    positions: Iterable[float],
    *,
    color: ColorType,
    size: float,
    arrowstyle: str,
    reverse: bool,
) -> list[Annotation]:
    """Draw direction arrows at fractional arc-length positions along a polyline.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    xdata : np.ndarray
        X values of the polyline.
    ydata : np.ndarray
        Y values of the polyline.
    positions : Iterable[float]
        Fractional positions along the polyline between 0 and 1.
    color : ColorType
        Arrow color.
    size : float
        Arrowhead size (mutation scale).
    arrowstyle : str
        Arrow style name.
    reverse : bool
        Point the arrows against the data order.

    Returns
    -------
    list[Annotation]
        The created arrow annotations.

    Raises
    ------
    ValueError
        If a position is outside [0, 1].

    """
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xdata), np.diff(ydata)))))
    annotations: list[Annotation] = []
    for position in positions:
        if not 0.0 <= position <= 1.0:
            msg = f"`positions` must be between 0 and 1: {position}"
            raise ValueError(msg)
        index = int(np.searchsorted(arc, position * arc[-1], side="right")) - 1
        index = min(max(index, 0), len(xdata) - 2)
        start = (float(xdata[index]), float(ydata[index]))
        end = (float(xdata[index + 1]), float(ydata[index + 1]))
        if reverse:
            start, end = end, start
        annotations.append(_make_arrow(ax, start, end, color=color, size=size, arrowstyle=arrowstyle))
    return annotations


def _arrows_at_crossings(
    ax: Axes,
    xdata: np.ndarray,
    ydata: np.ndarray,
    crossing: tuple[tuple[float, float], tuple[float, float]],
    *,
    color: ColorType,
    size: float,
    arrowstyle: str,
    reverse: bool,
    which: Literal["first", "all"],
    min_spacing: float,
    seen_crossings: list[tuple[float, float]] | None,
) -> list[Annotation]:
    """Draw direction arrows where a polyline crosses an auxiliary segment.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    xdata : np.ndarray
        X values of the polyline.
    ydata : np.ndarray
        Y values of the polyline.
    crossing : tuple[tuple[float, float], tuple[float, float]]
        End points of the auxiliary segment.
    color : ColorType
        Arrow color.
    size : float
        Arrowhead size (mutation scale).
    arrowstyle : str
        Arrow style name.
    reverse : bool
        Point the arrows against the data order.
    which : {"first", "all"}
        Whether to stop after the first crossing.
    min_spacing : float
        Crossings closer than this to an already processed crossing are skipped.
    seen_crossings : list[tuple[float, float]] or None
        Mutable record of processed crossings shared between calls.

    Returns
    -------
    list[Annotation]
        The created arrow annotations.

    """
    p2, q2 = crossing
    processed = seen_crossings if seen_crossings is not None else []
    annotations: list[Annotation] = []
    for i in range(len(xdata) - 1):
        p1 = (float(xdata[i]), float(ydata[i]))
        q1 = (float(xdata[i + 1]), float(ydata[i + 1]))
        if not _segments_intersect(p1, q1, p2, q2):
            continue
        if min_spacing > 0 and any(np.allclose(seen, p1, atol=min_spacing) for seen in processed):
            processed.append(p1)
            continue
        start, end = (q1, p1) if reverse else (p1, q1)
        annotations.append(_make_arrow(ax, start, end, color=color, size=size, arrowstyle=arrowstyle))
        processed.append(p1)
        if which == "first":
            break
    return annotations


def annotate_with_arrow(
    ax: Axes,
    text: str,
    xy: tuple[float, float],
    offset: tuple[float, float],
    *,
    fontsize: float | str | None = None,
    relpos: tuple[float, float] = (0.5, 0.5),
    pad: float = 0.15,
    lw: float = 0.6,
    color: ColorType = "black",
    text_color: ColorType | None = None,
    face_color: ColorType = "white",
    boxstyle: str = "square",
    arrowstyle: str = "->",
    shrink_a: float = 0.0,
    shrink_b: float = 0.8,
    **kwargs: Unknown,
) -> Annotation:
    """Annotate a plotted point with a boxed label and an arrow pointing at it.

    The label is placed at an offset in points from the annotated point, so its distance from
    the point is independent of the data scale.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    text : str
        Label text.
    xy : tuple[float, float]
        Point the arrow points at, in data coordinates.
    offset : tuple[float, float]
        Position of the label relative to `xy`, in points.
    fontsize : float or str or None, optional
        Font size of the label text, by default None (matplotlib default).
    relpos : tuple[float, float], optional
        Starting position of the arrow relative to the label box, by default (0.5, 0.5).
        (0, 0) is the lower left corner and (1, 1) is the upper right corner.
    pad : float, optional
        Margin inside the label box in fractions of the font size, by default 0.15.
    lw : float, optional
        Line width of the arrow and the label box edge, by default 0.6. Use 0 to hide the
        box edge.
    color : ColorType, optional
        Color of the arrow and the label box edge, by default "black".
    text_color : ColorType or None, optional
        Color of the label text, by default None, which follows `color`.
    face_color : ColorType, optional
        Background color of the label box, by default "white".
    boxstyle : str, optional
        Box style name of the label box, by default "square".
    arrowstyle : str, optional
        Arrow style name, by default "->".
    shrink_a : float, optional
        Gap between the label box and the arrow tail in points, by default 0.0.
    shrink_b : float, optional
        Gap between the arrow head and the annotated point in points, by default 0.8.
    **kwargs : Unknown
        Additional keyword arguments passed to `Axes.annotate`, e.g. `ha` (default "center")
        and `va` (default "bottom").

    Returns
    -------
    Annotation
        The created annotation.

    """
    bbox = {"boxstyle": f"{boxstyle},pad={pad}", "fc": face_color, "ec": color, "lw": lw}
    arrowprops = {
        "arrowstyle": arrowstyle,
        "relpos": relpos,
        "shrinkA": shrink_a,
        "shrinkB": shrink_b,
        "lw": lw,
        "color": color,
    }
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "bottom")
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    return ax.annotate(
        text,
        xy=xy,
        xytext=offset,
        textcoords="offset points",
        color=text_color if text_color is not None else color,
        bbox=bbox,
        arrowprops=arrowprops,
        **kwargs,
    )


# Local Variables:
# jinx-local-words: "Colormap FilePathT Iterable Jupyter LaTeX arg basename bbox ci cjk cmap csv customizable dataset ddof dir facecolor fmt ieee jp linspace lw matplotlib ndarray noqa np plt png randn sd se str timepoints timeseries tlim xlim ylim" # noqa: E501
# End:
