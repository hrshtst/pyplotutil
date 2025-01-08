from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np

from pyplotutil.loggingutil import evlog

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.typing import ColorType

    from pyplotutil._typing import FilePath


FilePathT = TypeVar("FilePathT", str, Path)


def compatible_filename(filename: FilePathT) -> FilePathT:
    table = {":": "", " ": "_", "(": "", ")": "", "+": "x", "=": "-"}
    compat_filename = str(filename).translate(str.maketrans(table))  # type: ignore[arg-type]
    if compat_filename != str(filename):
        evlog().debug("Filename has been converted to compatible one.")
        evlog().debug("      given filename: %s", filename)
        evlog().debug("  converted filename: %s", compat_filename)
    return type(filename)(compat_filename)


def make_figure_paths(
    output_direcotry: FilePath,
    basename: str,
    extensions: str | Iterable[str],
    *,
    separate_dir_by_main_module: bool | str,
    separate_dir_by_ext: bool,
) -> list[Path]:
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

    built_path = Path(output_direcotry)
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


def get_tlim_mask(t: np.ndarray, tlim: tuple[float, float] | None) -> np.ndarray:
    return np.full(t.shape, fill_value=True) if tlim is None else (t >= tlim[0]) & (t <= tlim[1])


def plot_multi_timeseries(
    ax: Axes,
    t: np.ndarray,
    data_array: np.ndarray,
    lw: int,
    c: str | None = None,
    labels: Iterable[str] | None = None,
    cmap_name: str = "tab10",
) -> None:
    data_array = np.atleast_2d(data_array)
    cmap = plt.get_cmap(cmap_name)
    if labels is None:
        labels = ["" for _ in range(len(data_array))]
    for i, (data, label) in enumerate(zip(data_array, labels, strict=True)):
        color = cmap(i) if c is None else c
        n = min(len(t), len(data))
        ax.plot(t[:n], data[:n], label=str(label), lw=lw, c=color)


def calculate_mean_err(
    data_array: np.ndarray,
    err_type: str = "std",
    ddof: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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


def plot_mean_err(
    ax: Axes,
    t: np.ndarray,
    y: np.ndarray,
    err_type: str | None,
    tlim: tuple[float, float] | None,
    fmt: str,
    capsize: int,
    label: str | None,
) -> Line2D:
    mask = get_tlim_mask(t, tlim)
    if err_type is None or err_type == "none":
        mean, _, _ = calculate_mean_err(y)
        lines = ax.plot(t[mask], mean[mask], fmt, label=label)
    else:
        mean, err1, err2 = calculate_mean_err(y, err_type=err_type)
        if err2 is None:
            eb = ax.errorbar(t[mask], mean[mask], yerr=err1[mask], capsize=capsize, fmt=fmt, label=label)
        else:
            eb = ax.errorbar(t[mask], mean[mask], yerr=(err1[mask], err2[mask]), capsize=capsize, fmt=fmt, label=label)
        lines = eb.lines  # type: ignore[assignment]
    return lines[0]


def fill_between_err(
    ax: Axes,
    t: np.ndarray,
    y: np.ndarray,
    err_type: str | None,
    tlim: tuple[float, float] | None,
    color: ColorType,
    alpha: float,
) -> Axes:
    if err_type is None:
        msg = "`err_type` for `fill_between_err` must not be None."
        raise TypeError(msg)

    mask = get_tlim_mask(t, tlim)
    mean, err1, err2 = calculate_mean_err(y, err_type=err_type)
    # Note that fill_between always goes behind lines.
    if err2 is None:
        ax.fill_between(t[mask], mean[mask] + err1[mask], mean[mask] - err1[mask], facecolor=color, alpha=alpha)
    else:
        ax.fill_between(t[mask], mean[mask] + err1[mask], mean[mask] - err2[mask], facecolor=color, alpha=alpha)
    return ax


# Local Variables:
# jinx-local-words: "arg ci csv dataset sd se"
# End:
