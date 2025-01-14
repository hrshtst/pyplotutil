"""Plotting Utilities for Scientific Data Visualization.

This module provides a collection of utilities for creating and saving scientific plots,
particularly focused on time series data visualization with error representations.

Key Features
-----------
- Save figures with multiple file formats
- Plot multiple time series with customizable styles
- Error visualization (standard deviation, variance, range, standard error)
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

from pyplotutil._typing import NoDefault, no_default
from pyplotutil.loggingutil import evlog

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
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
            import __main__

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Calculate mean and error metrics for data array.

    Parameters
    ----------
    data_array : np.ndarray
        Input data array.
    err_type : str, optional
        Type of error to calculate ("std", "var", "range", "se", "ci"), by default "std".
    ddof : int, optional
        Delta degrees of freedom, by default 0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
        Mean, error1, and error2 (if applicable) arrays.

    Raises
    ------
    TypeError
        If err_type is not a string.
    ValueError
        If err_type is not recognized.
    NotImplementedError
        If err_type is "ci".

    """
    if not isinstance(err_type, str):
        msg = f"`err_type` must be string: {err_type}, (type: {type(err_type)})"
        raise TypeError(msg)

    mean = np.mean(data_array, axis=0)
    if err_type.lower() in ("std", "sd"):
        # standard deviation
        std = np.std(data_array, axis=0, ddof=ddof)
        return mean, std, None

    if err_type.lower() in ("var",):
        # variance
        var = np.var(data_array, axis=0, ddof=ddof)
        return mean, var, None

    if err_type.lower() in ("range",):
        # range
        lower = mean - np.min(data_array, axis=0)
        upper = np.max(data_array, axis=0) - mean
        return mean, lower, upper

    if err_type.lower() in ("se",):
        # standard error
        std = np.std(data_array, axis=0, ddof=ddof)
        se = std / np.sqrt(len(std))
        return mean, se, None

    if err_type.lower() in ("ci",):
        # confidence interval
        raise NotImplementedError

    msg = f"unrecognized error type: {err_type}"
    raise ValueError(msg)


def plot_multi_timeseries(
    ax: Axes,
    t: np.ndarray,
    y_arr: np.ndarray,
    *,
    tlim: tuple[float, float] | None,
    lw: int | None,
    color: ColorType | None = None,
    fmt: str | None = None,
    labels: str | Iterable[str] | None = None,
    cmap_name: str | None = None,
) -> list[Line2D]:
    """Plot multiple time series on the same axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    t : np.ndarray
        Array of time values.
    y_arr : np.ndarray
        Array of y values.
    tlim : tuple[float, float] or None
        Time limits.
    lw : int or None
        Line width.
    color : ColorType or None, optional
        Line color, by default None.
    fmt : str or None, optional
        Format string, by default None.
    labels : str or Iterable[str] or None, optional
        Labels for lines, by default None.
    cmap_name : str or None, optional
        Colormap name, by default None.

    Returns
    -------
    list[Line2D]
        List of plotted lines.

    """
    mask = get_tlim_mask(t, tlim)
    y_arr = np.atleast_2d(y_arr)
    cmap = plt.get_cmap(cmap_name) if cmap_name is not None else None
    if labels is None:
        labels = [f"{i}" for i in range(len(y_arr))]
    elif isinstance(labels, str):
        labels = [labels] if len(y_arr) == 1 else [f"{labels}_{i}" for i in range(len(y_arr))]

    kwargs: dict[str, Unknown] = {}
    if lw is not None:
        kwargs["lw"] = lw
    if color is not None:
        kwargs["c"] = color

    t_mask = t[mask]
    lines: list[Line2D] = []
    for i, (y, label) in enumerate(zip(y_arr, labels, strict=True)):
        kwargs["label"] = str(label)
        if color is None and cmap is not None:
            kwargs["c"] = cmap(i)

        if fmt is None:  # noqa: SIM108
            _lines = ax.plot(t_mask, y[mask], **kwargs)
        else:
            _lines = ax.plot(t_mask, y[mask], fmt, **kwargs)
        lines.extend(_lines)
    return lines


def plot_mean_err(
    ax: Axes,
    t: np.ndarray,
    y_arr: np.ndarray,
    err_type: str | None,
    *,
    tlim: tuple[float, float] | None,
    lw: int | None,
    capsize: int | None,
    color: ColorType | None = None,
    fmt: str | None = None,
    label: str | None = None,
) -> Line2D:
    """Plot mean with error bars.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    t : np.ndarray
        Array of time values.
    y_arr : np.ndarray
        Array of y values.
    err_type : str or None
        Type of error to plot.
    tlim : tuple[float, float] or None
        Time limits.
    lw : int or None
        Line width.
    capsize : int or None
        Size of error bar caps.
    color : ColorType or None, optional
        Line color, by default None.
    fmt : str or None, optional
        Format string, by default None.
    label : str or None, optional
        Label for the plot, by default None.

    Returns
    -------
    Line2D
        The plotted line.

    """
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
        mean, err1, err2 = calculate_mean_err(y_arr, err_type=err_type)
        if err2 is None:
            eb = ax.errorbar(t[mask], mean[mask], yerr=err1[mask], **kwargs)
        else:
            eb = ax.errorbar(t[mask], mean[mask], yerr=(err1[mask], err2[mask]), **kwargs)
        lines = [eb.lines[0]]
    return lines[0]


def fill_between_err(
    ax: Axes,
    t: np.ndarray,
    y_arr: np.ndarray,
    err_type: str | None,
    *,
    tlim: tuple[float, float] | None,
    color: ColorType | None,
    alpha: float | None,
    interpolation: bool = False,
    suppress_exception: bool = False,
) -> Axes:
    """Fill between error bounds.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    t : np.ndarray
        Array of time values.
    y_arr : np.ndarray
        Array of y values.
    err_type : str or None
        Type of error to fill.
    tlim : tuple[float, float] or None
        Time limits.
    color : ColorType or None
        Fill color.
    alpha : float or None
        Fill transparency.
    interpolation : bool, optional
        Whether to use interpolation, by default False.
    suppress_exception : bool, optional
        Whether to suppress exceptions, by default False.

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

    mask = get_tlim_mask(t, tlim)
    y_arr = np.atleast_2d(y_arr)

    kwargs: dict[str, Unknown] = {}
    if color is not None:
        kwargs["facecolor"] = color
    if alpha is not None:
        kwargs["alpha"] = alpha
    kwargs["interpolation"] = interpolation

    mean, err1, err2 = calculate_mean_err(y_arr, err_type=err_type)
    # Note that fill_between always goes behind lines.
    if err2 is None:
        ax.fill_between(t[mask], mean[mask] + err1[mask], mean[mask] - err1[mask], **kwargs)
    else:
        ax.fill_between(t[mask], mean[mask] + err1[mask], mean[mask] - err2[mask], **kwargs)
    return ax


# Local Variables:
# jinx-local-words: "Colormap FilePathT Iterable Jupyter LaTeX arg basename bbox ci cjk cmap csv customizable dataset ddof dir facecolor fmt ieee jp linspace lw matplotlib ndarray noqa np plt png randn sd se str timepoints timeseries tlim xlim ylim" # noqa: E501
# End:
