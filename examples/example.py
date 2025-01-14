#!/usr/bin/env python

"""Example script of pyplotutil module."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from pyplotutil.datautil import Data, Dataset, TaggedData
from pyplotutil.loggingutil import evlog, start_logging
from pyplotutil.plotutil import apply_style, fill_between_err, plot_mean_err, plot_multi_timeseries, save_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


pparam = {"xlabel": "Voltage (mV)", "ylabel": r"Current ($\mu$A)"}


def load_data(data: str, *, tagged: bool) -> Data | TaggedData | Dataset:
    """Load data from provided file paths."""
    data_path = Path(data)
    if data_path.is_dir():
        return Dataset(data)
    if tagged:
        return TaggedData(data)
    return Data(data)


def plot_data(data: Data, output_dir: Path, ext: list[str] | None) -> tuple[Figure, Axes]:
    """Plot a single data object."""
    evlog().debug("Plot a Data object: %s", repr(data))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data.x, data.y, label=int(data.param("p")))
    ax.legend(title="Order")
    ax.autoscale(tight=True)
    ax.set(**pparam)
    basename = data.datapath.stem if data.is_loaded_from_file() else "data"
    save_figure(fig, output_dir, basename, ext)
    return fig, ax


def plot_tagged_data(tagged_data: TaggedData, output_dir: Path, ext: list[str] | None) -> tuple[Figure, Axes]:
    """Plot all data contained in a tagged data object."""
    evlog().debug("Plot a TaggedData object: %s", repr(tagged_data))
    fig, ax = plt.subplots(figsize=(8, 6))
    for tag, data in tagged_data:
        ax.plot(data.x, data.y, label=tagged_data.param("p", tag=tag))
    ax.legend(title="Order")
    ax.autoscale(tight=True)
    ax.set(**pparam)
    basename = tagged_data.datapath.stem if tagged_data.is_loaded_from_file() else "tagged_data"
    save_figure(fig, output_dir, basename, ext)
    return fig, ax


def plot_dataset(
    dataset: Dataset,
    output_dir: Path,
    ext: list[str] | None,
    *,
    err_type: str | None,
) -> tuple[Figure, Axes]:
    """Plot all data contained in a dataset object."""
    evlog().debug("Plot a Dataset object: %s", repr(dataset))
    fig, ax = plt.subplots(figsize=(8, 6))
    x, y = dataset.get_axis_data("x", "y")
    if err_type is None:
        labels = map(str, map(int, [data.param("p") for data in dataset]))
        plot_multi_timeseries(ax, x, y, tlim=None, lw=1, labels=labels)
    else:
        plot_mean_err(ax, x, y, None, tlim=None, lw=1, capsize=None, label="mean")
        fill_between_err(ax, x, y, err_type, tlim=None, color=None, alpha=0.2)
    ax.legend(title="Order")
    ax.autoscale(tight=True)
    ax.set(**pparam)
    basename = f"models_{err_type}" if err_type is not None else "dataset"
    save_figure(fig, output_dir, basename, ext)
    return fig, ax


def parse() -> argparse.Namespace:
    """Parse provided command arguments."""
    parser = argparse.ArgumentParser(description="Compare scores across preview/delay steps")
    parser.add_argument("data", help="Path to a CSV file or a directory containing CSV files.")
    parser.add_argument("-o", "--output-directory", dest="output_dir", default=".", help="Output directory.")
    parser.add_argument("-e", "--extension", dest="ext", nargs="+", help="Output file extension.")
    parser.add_argument("--err-type", help="Error type for dataset plot.")
    parser.add_argument(
        "--tagged",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load a data file as tagged data.",
    )
    parser.add_argument("--style", choices=["science", "ieee", "nature", "notebook"], help="Plot style.")
    parser.add_argument(
        "--grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boolean. If True, show the axis grids in the figure. (default: False)",
    )
    parser.add_argument(
        "--latex",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boolean. If True, render the figure without latex. (default: False)",
    )
    parser.add_argument(
        "--show-screen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Boolean. If True, show the plot figure. (default: True)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose output. -v provides additional info. -vv provides debug output.",
    )
    return parser.parse_args()


def main() -> None:
    """Run main function."""
    import sys

    args = parse()
    start_logging(sys.argv, args.output_dir, __name__, args.verbose)
    evlog().info("Output directory: %s", args.output_dir)
    evlog().debug("Parsed arguments: %s", args)

    data = load_data(args.data, tagged=args.tagged)
    evlog().info("Loaded data file: %s", args.data)

    output_dir = Path(args.output_dir)
    if args.style is not None:
        evlog().debug("Apply plot style: %s", args.style)
        apply_style(args.style, grid=args.grid, no_latex=not args.latex)
        output_dir /= args.style
    elif args.grid:
        plt.rcParams["axes.grid"] = True

    if isinstance(data, TaggedData):
        plot_tagged_data(data, output_dir, args.ext)
    elif isinstance(data, Dataset):
        plot_dataset(data, output_dir, args.ext, err_type=args.err_type)
    else:
        plot_data(data, output_dir, args.ext)

    if args.show_screen:
        plt.show()


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "dataset dir env ieee mV pyplotutil usr vv xlabel ylabel"
# End:
