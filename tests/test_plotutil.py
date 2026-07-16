# ruff: noqa: S101
"""Unit tests for plotting utility functions.

This test suite covers:
- Filename compatibility conversion for both string and Path objects
- Figure path generation with various configurations:
    * Single and multiple extensions
    * Directory separation by main module
    * Directory separation by extension
    * Duplicate extension handling
- Common path extraction functionality
- Style application and rcParams updates
- Mean and error calculation for every error type
- Time series and mean-with-error plotting
- Error band filling and figure saving

The tests use parametrized fixtures to verify multiple input scenarios and edge cases for each function.

"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from pyplotutil.datautil import Dataset, TaggedData
from pyplotutil.plotutil import (
    add_direction_arrows,
    annotate_with_arrow,
    apply_style,
    calculate_mean_err,
    compatible_filename,
    extract_common_path,
    fill_between_err,
    get_limits,
    get_tlim_mask,
    label_with_unit,
    make_figure_paths,
    mask_to_spans,
    plot_mean_err,
    plot_multi_timeseries,
    save_figure,
    setup_axes,
    shade_spans,
)
from tests.test_datautil import DATA_DIR_PATH

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

    from pyplotutil._typing import FilePath


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("file.dat", "file.dat"),
        ("./space file.dat", "./space_file.dat"),
        ("a/colon: file.dat", "a/colon_file.dat"),
        ("(parenthesis) file.dat", "parenthesis_file.dat"),
        ("a+b/c++.dat", "axb/cxx.dat"),
        ("/a/b/c/alpha=0.1.dat", "/a/b/c/alpha-0.1.dat"),
        ("title: alpha=1.0 (beta=0.1).dat", "title_alpha-1.0_beta-0.1.dat"),
    ],
)
def test_compatible_filename(filename: str, expected: str) -> None:
    """Test string filename compatibility conversion."""
    converted = compatible_filename(filename)
    assert type(converted) is str
    assert converted == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("file.dat", "file.dat"),
        ("./space file.dat", "./space_file.dat"),
        ("a/colon: file.dat", "a/colon_file.dat"),
        ("(parenthesis) file.dat", "parenthesis_file.dat"),
        ("a+b/c++.dat", "axb/cxx.dat"),
        ("/a/b/c/alpha=0.1.dat", "/a/b/c/alpha-0.1.dat"),
        ("title: alpha=1.0 (beta=0.1).dat", "title_alpha-1.0_beta-0.1.dat"),
    ],
)
def test_compatible_filename_path(filename: str, expected: str) -> None:
    """Test Path object filename compatibility conversion."""
    converted = compatible_filename(Path(filename))
    assert isinstance(converted, Path)
    assert converted == Path(expected)


@pytest.mark.parametrize(
    ("output_directory", "basename", "extension", "expected"),
    [
        (".", "figure", ".png", "./figure.png"),
        ("", "data.dat", ".png", "./data.png"),
        ("figure", "data.dat", "png", "figure/data.png"),
        ("figure", "data", "png", "figure/data.png"),
        ("output/fig", "alpha-0.1.dat", "png", "output/fig/alpha-0.1.png"),
        ("output/fig", "alpha-0.1", "png", "output/fig/alpha-0.1.png"),
        ("p-p+", "title: alpha=1.0 (beta=0.1).dat", ".png", "p-px/title_alpha-1.0_beta-0.1.png"),
        ("p-p+", "title: alpha=1.0 (beta=0.1)", ".png", "p-px/title_alpha-1.0_beta-0.1.png"),
    ],
)
def test_make_figure_paths_single_ext(
    output_directory: FilePath,
    basename: str,
    extension: str,
    expected: str,
) -> None:
    """Test figure path generation with single extension."""
    figure_paths = make_figure_paths(
        output_directory,
        basename,
        extension,
        separate_dir_by_main_module=False,
        separate_dir_by_ext=False,
    )
    assert figure_paths == [Path(expected)]


@pytest.mark.parametrize(
    ("output_directory", "basename", "extensions", "expected"),
    [
        (".", "figure", [".png"], ["./figure.png"]),
        ("figure", "data.dat", [".png", ".pdf"], ["figure/data.png", "figure/data.pdf"]),
        ("figure", "data.dat", [".png", ".pdf", ".svg"], ["figure/data.png", "figure/data.pdf", "figure/data.svg"]),
        ("figure", "data.dat", ["png", "pdf"], ["figure/data.png", "figure/data.pdf"]),
        ("output/fig", "alpha-0.1.dat", ["png", ".svg"], ["output/fig/alpha-0.1.png", "output/fig/alpha-0.1.svg"]),
        ("output/fig", "alpha-0.1", ["png", "pdf"], ["output/fig/alpha-0.1.png", "output/fig/alpha-0.1.pdf"]),
        (
            "p-p+",
            "title: alpha=1.0 (beta=0.1).dat",
            [".pdf", ".png"],
            ["p-px/title_alpha-1.0_beta-0.1.pdf", "p-px/title_alpha-1.0_beta-0.1.png"],
        ),
        (
            "p-p+",
            "title: alpha=1.0 (beta=0.1)",
            [".png", ".jpg"],
            ["p-px/title_alpha-1.0_beta-0.1.png", "p-px/title_alpha-1.0_beta-0.1.jpg"],
        ),
    ],
)
def test_make_figure_paths_multiple_ext(
    output_directory: FilePath,
    basename: str,
    extensions: list[str],
    expected: list[str],
) -> None:
    """Test figure path generation with multiple extensions."""
    figure_paths = make_figure_paths(
        output_directory,
        basename,
        extensions,
        separate_dir_by_main_module=False,
        separate_dir_by_ext=False,
    )
    assert set(figure_paths) == {Path(e) for e in expected}


@pytest.mark.parametrize(
    ("output_directory", "basename", "extensions", "expected"),
    [
        ("figure", "data.dat", [".png", ".pdf", ".pdf"], ["figure/data.png", "figure/data.pdf"]),
        ("figure", "data.dat", [".pdf", ".pdf", ".pdf"], ["figure/data.pdf"]),
        (
            "figure",
            "data.dat",
            ["svg", ".png", ".pdf", ".png", ".pdf"],
            ["figure/data.svg", "figure/data.png", "figure/data.pdf"],
        ),
    ],
)
def test_make_figure_paths_remove_duplicates(
    output_directory: FilePath,
    basename: str,
    extensions: list[str],
    expected: list[str],
) -> None:
    """Test duplicate extension removal in figure path generation."""
    figure_paths = make_figure_paths(
        output_directory,
        basename,
        extensions,
        separate_dir_by_main_module=False,
        separate_dir_by_ext=False,
    )
    assert set(figure_paths) == {Path(e) for e in expected}


@pytest.mark.parametrize(
    ("output_directory", "basename", "extensions", "main_module", "expected"),
    [
        ("figure", "data.dat", [".png", ".pdf"], False, ["figure/data.png", "figure/data.pdf"]),
        ("figure", "data.dat", [".png", ".pdf"], True, ["figure/pytest/data.png", "figure/pytest/data.pdf"]),
        (
            "figure",
            "data.dat",
            [".png", ".pdf"],
            "pyplotutil",
            ["figure/pyplotutil/data.png", "figure/pyplotutil/data.pdf"],
        ),
    ],
)
def test_make_figure_paths_separate_dir_by_main_module(
    *,
    output_directory: FilePath,
    basename: str,
    extensions: list[str],
    main_module: bool | str,
    expected: list[str],
) -> None:
    """Test figure path generation with main module directory separation."""
    figure_paths = make_figure_paths(
        output_directory,
        basename,
        extensions,
        separate_dir_by_main_module=main_module,
        separate_dir_by_ext=False,
    )
    assert set(figure_paths) == {Path(e) for e in expected}


@pytest.mark.parametrize(
    ("output_directory", "basename", "extensions", "expected"),
    [
        ("figure", "data.dat", [".png", ".pdf"], ["figure/png/data.png", "figure/pdf/data.pdf"]),
        ("figure", "data.dat", [".png", ".pdf", ".pdf"], ["figure/png/data.png", "figure/pdf/data.pdf"]),
        ("figure", "data.1", ["png", "pdf"], ["figure/png/data.1.png", "figure/pdf/data.1.pdf"]),
        ("figure", "data.1.dat", ["png", ".pdf", "pdf"], ["figure/png/data.1.png", "figure/pdf/data.1.pdf"]),
    ],
)
def test_make_figure_paths_separate_dir_by_ext(
    output_directory: FilePath,
    basename: str,
    extensions: list[str],
    expected: list[str],
) -> None:
    """Test figure path generation with extension directory separation."""
    figure_paths = make_figure_paths(
        output_directory,
        basename,
        extensions,
        separate_dir_by_main_module=False,
        separate_dir_by_ext=True,
    )
    assert set(figure_paths) == {Path(e) for e in expected}


@pytest.mark.parametrize(
    ("output_directory", "basename", "extensions", "main_module", "separate_dir_by_ext", "expected"),
    [
        (
            "figure",
            "data.dat",
            [".png", ".pdf"],
            True,
            True,
            ["figure/pytest/png/data.png", "figure/pytest/pdf/data.pdf"],
        ),
        (
            "./output",
            "plot (alpha=1.5)",
            ["png", "svg"],
            "pyplotutil",
            True,
            ["./output/pyplotutil/png/plot_alpha-1.5.png", "./output/pyplotutil/svg/plot_alpha-1.5.svg"],
        ),
    ],
)
def test_make_figure_paths(
    *,
    output_directory: FilePath,
    basename: str,
    extensions: list[str],
    main_module: bool | str,
    separate_dir_by_ext: bool,
    expected: list[str],
) -> None:
    """Test comprehensive figure path generation with all options."""
    figure_paths = make_figure_paths(
        output_directory,
        basename,
        extensions,
        separate_dir_by_main_module=main_module,
        separate_dir_by_ext=separate_dir_by_ext,
    )
    assert set(figure_paths) == {Path(e) for e in expected}


IEEE_DPI = 100.0
NOTEBOOK_FIGSIZE = [8.0, 6.0]


@pytest.mark.parametrize(
    ("style", "check"),
    [
        ("science", lambda: plt.rcParams["text.usetex"] is True),
        ("ieee", lambda: plt.rcParams["figure.dpi"] == IEEE_DPI),
        ("nature", lambda: plt.rcParams["text.usetex"] is True),
        ("notebook", lambda: plt.rcParams["figure.figsize"] == NOTEBOOK_FIGSIZE),
    ],
)
def test_apply_style(style: str, check: Callable[[], bool]) -> None:
    """Test that each style name updates the expected matplotlib rcParams."""
    with mpl.rc_context():
        apply_style(style)  # type: ignore[arg-type]
        assert check()


def test_apply_style_options() -> None:
    """Test that style options toggle the corresponding rcParams."""
    with mpl.rc_context():
        apply_style("science", grid=True, no_latex=True)
        assert plt.rcParams["axes.grid"] is True
        assert plt.rcParams["text.usetex"] is False


def test_apply_style_unsupported() -> None:
    """Test that an unsupported style name raises ValueError."""
    with mpl.rc_context(), pytest.raises(ValueError, match="Unsupported style: fancy"):
        apply_style("fancy")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("tlim", "expected"),
    [
        (None, [True, True, True, True]),
        ((1.0, 2.0), [False, True, True, False]),
        ((0.0, 0.5), [True, False, False, False]),
        ((4.0, 9.0), [False, False, False, False]),
    ],
)
def test_get_tlim_mask(tlim: tuple[float, float] | None, expected: list[bool]) -> None:
    """Test boolean mask generation for time limits."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(get_tlim_mask(t, tlim), np.array(expected))


class TestCalculateMeanErr:
    """A class collecting tests for `calculate_mean_err`."""

    data_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 12.0]])

    @pytest.mark.parametrize("err_type", ["std", "sd", "STD", "Sd"])
    def test_std(self, err_type: str) -> None:
        """Test standard deviation error, including case-insensitive aliases."""
        mean, err1, err2 = calculate_mean_err(self.data_array, err_type=err_type)
        np.testing.assert_allclose(mean, np.mean(self.data_array, axis=0))
        np.testing.assert_allclose(err1, np.std(self.data_array, axis=0))
        assert err2 is None

    def test_std_ddof(self) -> None:
        """Test that ddof is forwarded to the deviation calculation."""
        _, err1, _ = calculate_mean_err(self.data_array, err_type="std", ddof=1)
        np.testing.assert_allclose(err1, np.std(self.data_array, axis=0, ddof=1))

    def test_var(self) -> None:
        """Test variance error."""
        mean, err1, err2 = calculate_mean_err(self.data_array, err_type="var")
        np.testing.assert_allclose(mean, np.mean(self.data_array, axis=0))
        np.testing.assert_allclose(err1, np.var(self.data_array, axis=0))
        assert err2 is None

    def test_range(self) -> None:
        """Test that range errors are distances from the mean to the extremes."""
        mean, err1, err2 = calculate_mean_err(self.data_array, err_type="range")
        np.testing.assert_allclose(err1, mean - np.min(self.data_array, axis=0))
        assert err2 is not None
        np.testing.assert_allclose(err2, np.max(self.data_array, axis=0) - mean)

    # Student's t critical values for the two-sided 95% and 99% levels with two degrees
    # of freedom (three trials).
    T_CRIT_95_DF2 = 4.302653
    T_CRIT_99_DF2 = 9.924843

    def test_ci(self) -> None:
        """Test that the confidence interval scales the standard error by the t critical value."""
        mean, err1, err2 = calculate_mean_err(self.data_array, err_type="ci")
        np.testing.assert_allclose(mean, np.mean(self.data_array, axis=0))
        se = np.std(self.data_array, axis=0) / np.sqrt(self.data_array.shape[0])
        np.testing.assert_allclose(err1, self.T_CRIT_95_DF2 * se, rtol=1e-6)
        assert err2 is None

    def test_ci_confidence_level(self) -> None:
        """Test that the confidence level changes the critical value."""
        _, err1, _ = calculate_mean_err(self.data_array, err_type="ci", confidence=0.99)
        se = np.std(self.data_array, axis=0) / np.sqrt(self.data_array.shape[0])
        np.testing.assert_allclose(err1, self.T_CRIT_99_DF2 * se, rtol=1e-6)

    def test_ci_ddof(self) -> None:
        """Test that ddof is applied to the deviation underlying the confidence interval."""
        _, err1, _ = calculate_mean_err(self.data_array, err_type="ci", ddof=1)
        se = np.std(self.data_array, axis=0, ddof=1) / np.sqrt(self.data_array.shape[0])
        np.testing.assert_allclose(err1, self.T_CRIT_95_DF2 * se, rtol=1e-6)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5])
    def test_ci_invalid_confidence(self, confidence: float) -> None:
        """Test that a confidence level outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="`confidence` must be between 0 and 1"):
            calculate_mean_err(self.data_array, err_type="ci", confidence=confidence)

    def test_ci_requires_two_trials(self) -> None:
        """Test that a confidence interval with a single trial raises ValueError."""
        with pytest.raises(ValueError, match="requires at least 2 trials"):
            calculate_mean_err(np.array([[1.0, 2.0]]), err_type="ci")

    def test_unrecognized_err_type(self) -> None:
        """Test that an unknown error type raises ValueError."""
        with pytest.raises(ValueError, match="unrecognized error type: bogus"):
            calculate_mean_err(self.data_array, err_type="bogus")

    def test_non_string_err_type(self) -> None:
        """Test that a non-string error type raises TypeError."""
        with pytest.raises(TypeError, match="`err_type` must be string"):
            calculate_mean_err(self.data_array, err_type=123)  # type: ignore[arg-type]


class TestPlotMultiTimeseries:
    """A class collecting tests for `plot_multi_timeseries`."""

    t = np.linspace(0.0, 1.0, 11)
    y_arr = np.vstack([np.sin(t), np.cos(t)])

    def test_default_labels(self) -> None:
        """Test that lines are labeled by index when no labels are given."""
        ax = Figure().add_subplot()
        lines = plot_multi_timeseries(ax, self.t, self.y_arr, tlim=None, lw=None)
        assert [line.get_label() for line in lines] == ["0", "1"]

    def test_string_label_expansion(self) -> None:
        """Test that a single string label is expanded per line."""
        ax = Figure().add_subplot()
        lines = plot_multi_timeseries(ax, self.t, self.y_arr, tlim=None, lw=None, labels="y")
        assert [line.get_label() for line in lines] == ["y_0", "y_1"]

    def test_string_label_single_series(self) -> None:
        """Test that a single string label is kept as-is for a single series."""
        ax = Figure().add_subplot()
        lines = plot_multi_timeseries(ax, self.t, self.y_arr[0], tlim=None, lw=None, labels="y")
        assert [line.get_label() for line in lines] == ["y"]

    def test_tlim_masks_data(self) -> None:
        """Test that time limits clip the plotted data."""
        ax = Figure().add_subplot()
        tlim = (0.2, 0.8)
        lines = plot_multi_timeseries(ax, self.t, self.y_arr, tlim=tlim, lw=None)
        for line in lines:
            xdata = np.asarray(line.get_xdata())
            assert xdata.min() >= tlim[0]
            assert xdata.max() <= tlim[1]

    def test_cmap_assigns_distinct_colors(self) -> None:
        """Test that a colormap assigns a distinct color per line."""
        ax = Figure().add_subplot()
        lines = plot_multi_timeseries(ax, self.t, self.y_arr, tlim=None, lw=None, cmap_name="viridis")
        assert lines[0].get_color() != lines[1].get_color()

    def test_fmt_and_lw(self) -> None:
        """Test that format string and line width are applied."""
        ax = Figure().add_subplot()
        line_width = 3
        lines = plot_multi_timeseries(ax, self.t, self.y_arr, tlim=None, lw=line_width, fmt="--")
        assert all(line.get_linewidth() == line_width for line in lines)


class TestPlotMeanErr:
    """A class collecting tests for `plot_mean_err`."""

    t = np.linspace(0.0, 1.0, 11)
    y_arr = np.vstack([np.sin(t), np.cos(t), np.sin(t) + 1.0])

    def test_no_error(self) -> None:
        """Test that plotting without error draws the mean only."""
        ax = Figure().add_subplot()
        line = plot_mean_err(ax, self.t, self.y_arr, None, tlim=None, lw=None, capsize=None, label="mean")
        np.testing.assert_allclose(np.asarray(line.get_ydata()), np.mean(self.y_arr, axis=0))
        assert not ax.containers

    def test_symmetric_error(self) -> None:
        """Test that a one-sided error type draws error bars around the mean."""
        ax = Figure().add_subplot()
        line = plot_mean_err(ax, self.t, self.y_arr, "std", tlim=None, lw=None, capsize=2)
        np.testing.assert_allclose(np.asarray(line.get_ydata()), np.mean(self.y_arr, axis=0))
        assert len(ax.containers) == 1

    def test_two_sided_error(self) -> None:
        """Test that a two-sided error type draws asymmetric error bars."""
        ax = Figure().add_subplot()
        line = plot_mean_err(ax, self.t, self.y_arr, "range", tlim=None, lw=1, capsize=None, color="C1")
        np.testing.assert_allclose(np.asarray(line.get_ydata()), np.mean(self.y_arr, axis=0))
        assert len(ax.containers) == 1

    def test_confidence_is_forwarded(self) -> None:
        """Test that the confidence level reaches the error calculation."""
        ax = Figure().add_subplot()
        plot_mean_err(ax, self.t, self.y_arr, "ci", confidence=0.99)
        barlinecols = ax.containers[0][2][0]
        mean = np.mean(self.y_arr, axis=0)
        se = np.std(self.y_arr, axis=0) / np.sqrt(self.y_arr.shape[0])
        expected = TestCalculateMeanErr.T_CRIT_99_DF2 * se
        for i, segment in enumerate(barlinecols.get_segments()):
            assert segment[:, 1].min() == pytest.approx(mean[i] - expected[i], rel=1e-6)
            assert segment[:, 1].max() == pytest.approx(mean[i] + expected[i], rel=1e-6)


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Dataset:
    """Create a Dataset of three CSV files where column a of file i equals t + i."""
    for i in range(3):
        rows = "\n".join(f"{t},{t + i}" for t in range(5))
        (tmp_path / f"data{i}.csv").write_text(f"t,a\n{rows}\n")
    return Dataset(tmp_path)


class TestPlotFromDataset:
    """A class collecting tests for plotting directly from a Dataset."""

    def test_plot_multi_timeseries(self, sample_dataset: Dataset) -> None:
        """Test that a Dataset plots one line per data file, labeled by file stem."""
        ax = Figure().add_subplot()
        lines = plot_multi_timeseries(ax, sample_dataset, "a")
        assert [line.get_label() for line in lines] == ["data0", "data1", "data2"]
        for i, line in enumerate(lines):
            np.testing.assert_allclose(np.asarray(line.get_ydata()), np.arange(5) + i)

    def test_plot_multi_timeseries_t_shift(self, sample_dataset: Dataset) -> None:
        """Test that the time shift moves the x values of every line."""
        ax = Figure().add_subplot()
        lines = plot_multi_timeseries(ax, sample_dataset, "a", t_shift=1.0)
        np.testing.assert_allclose(np.asarray(lines[0].get_xdata()), np.arange(5) - 1.0)

    def test_plot_mean_err(self, sample_dataset: Dataset) -> None:
        """Test that the mean across data files is plotted with error bars."""
        ax = Figure().add_subplot()
        line = plot_mean_err(ax, sample_dataset, "a", "std", label="mean")
        np.testing.assert_allclose(np.asarray(line.get_ydata()), np.arange(5) + 1.0)
        assert len(ax.containers) == 1
        assert ax.containers[0].get_label() == "mean"

    def test_fill_between_err(self, sample_dataset: Dataset) -> None:
        """Test that the range band spans the per-file extremes of the column."""
        ax = Figure().add_subplot()
        fill_between_err(ax, sample_dataset, "a", "range")
        vertices = np.asarray(ax.collections[0].get_paths()[0].vertices)
        for x in range(5):
            band_y = vertices[np.isclose(vertices[:, 0], x), 1]
            assert band_y.min() == pytest.approx(x)
            assert band_y.max() == pytest.approx(x + 2)


class TestPlotFromTaggedData:
    """A class collecting tests for plotting directly from a TaggedData."""

    CSV = "tag,t,a\nx,0,1\nx,1,2\ny,0,3\ny,1,5\n"

    def test_one_line_per_tag(self) -> None:
        """Test that each tag group is plotted as its own labeled line."""
        tagged = TaggedData(StringIO(self.CSV))
        ax = Figure().add_subplot()
        lines = plot_multi_timeseries(ax, tagged, "a")
        assert [line.get_label() for line in lines] == ["x", "y"]
        np.testing.assert_allclose(np.asarray(lines[0].get_ydata()), [1.0, 2.0])
        np.testing.assert_allclose(np.asarray(lines[1].get_ydata()), [3.0, 5.0])

    def test_cmap_assigns_distinct_colors(self) -> None:
        """Test that a colormap assigns a distinct color per tag."""
        tagged = TaggedData(StringIO(self.CSV))
        ax = Figure().add_subplot()
        lines = plot_multi_timeseries(ax, tagged, "a", cmap_name="viridis")
        assert lines[0].get_color() != lines[1].get_color()


class TestFillBetweenErr:
    """A class collecting tests for `fill_between_err`."""

    t = np.linspace(0.0, 1.0, 11)
    y_arr = np.vstack([np.sin(t), np.cos(t), np.sin(t) + 1.0])

    def test_none_err_type_raises(self) -> None:
        """Test that a missing error type raises ValueError by default."""
        ax = Figure().add_subplot()
        with pytest.raises(ValueError, match="must not be None"):
            fill_between_err(ax, self.t, self.y_arr, None, tlim=None, color=None, alpha=None)

    def test_none_err_type_suppressed(self) -> None:
        """Test that a missing error type is a no-op when suppressed."""
        ax = Figure().add_subplot()
        result = fill_between_err(
            ax, self.t, self.y_arr, None, tlim=None, color=None, alpha=None, suppress_exception=True
        )
        assert result is ax
        assert not ax.collections

    def test_symmetric_band(self) -> None:
        """Test that a one-sided error type fills the band mean +/- err."""
        ax = Figure().add_subplot()
        fill_between_err(ax, self.t, self.y_arr, "std", tlim=None, color="C0", alpha=0.3)
        mean = np.mean(self.y_arr, axis=0)
        std = np.std(self.y_arr, axis=0)
        vertices = np.asarray(ax.collections[0].get_paths()[0].vertices)
        for i, x in enumerate(self.t):
            band_y = vertices[np.isclose(vertices[:, 0], x), 1]
            assert band_y.min() == pytest.approx(mean[i] - std[i])
            assert band_y.max() == pytest.approx(mean[i] + std[i])


@pytest.mark.parametrize(
    ("label", "unit", "units", "expected"),
    [
        ("Position", "m", None, "Position [m]"),
        ("Position", "[m]", None, "Position [m]"),
        ("Force", None, {"force": "N"}, "Force [N]"),
        ("Force", "kN", {"force": "N"}, "Force [kN]"),
        ("Phase", None, {"force": "N"}, "Phase"),
        ("Position", None, None, "Position"),
        (None, "m", None, ""),
    ],
)
def test_label_with_unit(
    label: str | None,
    unit: str | None,
    units: dict[str, str] | None,
    expected: str,
) -> None:
    """Test label composition with explicit units, unit lookup, and fallbacks."""
    assert label_with_unit(label, unit, units=units) == expected


class TestAddDirectionArrows:
    """A class collecting tests for `add_direction_arrows`."""

    @staticmethod
    def straight_line(ax: Axes) -> Line2D:
        """Plot a straight line along the x axis with unit-spaced points."""
        x = np.arange(11.0)
        return ax.plot(x, np.zeros_like(x), color="C0")[0]

    @staticmethod
    def zigzag_line(ax: Axes) -> Line2D:
        """Plot a line crossing y=0 twice: rising at x=0-1 and falling at x=1-2."""
        return ax.plot([0.0, 1.0, 2.0, 3.0], [-1.0, 1.0, -1.0, -1.0], color="C0")[0]

    def test_default_places_one_arrow_at_midpoint(self) -> None:
        """Test that the default places a single arrow halfway along the line."""
        ax = Figure().add_subplot()
        line = self.straight_line(ax)
        annotations = add_direction_arrows(ax, line)
        assert len(annotations) == 1
        assert annotations[0].xyann == (5.0, 0.0)
        assert annotations[0].xy == (6.0, 0.0)

    def test_positions(self) -> None:
        """Test arrows at explicit fractional positions."""
        ax = Figure().add_subplot()
        line = self.straight_line(ax)
        annotations = add_direction_arrows(ax, line, positions=[0.0, 0.5, 1.0])
        assert [a.xyann for a in annotations] == [(0.0, 0.0), (5.0, 0.0), (9.0, 0.0)]
        assert [a.xy for a in annotations] == [(1.0, 0.0), (6.0, 0.0), (10.0, 0.0)]

    def test_reverse(self) -> None:
        """Test that reverse points the arrow against the data order."""
        ax = Figure().add_subplot()
        line = self.straight_line(ax)
        annotations = add_direction_arrows(ax, line, positions=[0.5], reverse=True)
        assert annotations[0].xyann == (6.0, 0.0)
        assert annotations[0].xy == (5.0, 0.0)

    def test_color_defaults_to_line_color(self) -> None:
        """Test that arrows follow the line color unless overridden."""
        ax = Figure().add_subplot()
        line = self.straight_line(ax)
        annotations = add_direction_arrows(ax, line)
        assert annotations[0].arrow_patch is not None
        assert annotations[0].arrow_patch.get_edgecolor() == mpl.colors.to_rgba("C0")

    def test_position_out_of_range(self) -> None:
        """Test that a position outside [0, 1] raises ValueError."""
        ax = Figure().add_subplot()
        line = self.straight_line(ax)
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            add_direction_arrows(ax, line, positions=[1.5])

    def test_mutually_exclusive_modes(self) -> None:
        """Test that giving both positions and crossing raises ValueError."""
        ax = Figure().add_subplot()
        line = self.straight_line(ax)
        with pytest.raises(ValueError, match="mutually exclusive"):
            add_direction_arrows(ax, line, positions=[0.5], crossing=((0.0, -1.0), (0.0, 1.0)))

    def test_too_few_points(self) -> None:
        """Test that a line with a single point raises ValueError."""
        ax = Figure().add_subplot()
        line = ax.plot([0.0], [0.0])[0]
        with pytest.raises(ValueError, match="at least 2 points"):
            add_direction_arrows(ax, line)

    def test_crossing_all(self) -> None:
        """Test arrows at every crossing with an auxiliary segment."""
        ax = Figure().add_subplot()
        line = self.zigzag_line(ax)
        annotations = add_direction_arrows(ax, line, crossing=((-10.0, 0.0), (10.0, 0.0)))
        expected_crossings = 2
        assert len(annotations) == expected_crossings
        assert annotations[0].xyann == (0.0, -1.0)
        assert annotations[0].xy == (1.0, 1.0)
        assert annotations[1].xyann == (1.0, 1.0)
        assert annotations[1].xy == (2.0, -1.0)

    def test_crossing_first_only(self) -> None:
        """Test that which='first' stops after the first crossing."""
        ax = Figure().add_subplot()
        line = self.zigzag_line(ax)
        annotations = add_direction_arrows(ax, line, crossing=((-10.0, 0.0), (10.0, 0.0)), which="first")
        assert len(annotations) == 1
        assert annotations[0].xyann == (0.0, -1.0)

    def test_seen_crossings_deduplicates_across_calls(self) -> None:
        """Test that a shared crossing record suppresses overlapping arrows."""
        ax = Figure().add_subplot()
        line = self.zigzag_line(ax)
        seen: list[tuple[float, float]] = []
        first = add_direction_arrows(ax, line, crossing=((-10.0, 0.0), (10.0, 0.0)), seen_crossings=seen)
        expected_crossings = 2
        assert len(first) == expected_crossings
        assert len(seen) == expected_crossings
        second = add_direction_arrows(ax, line, crossing=((-10.0, 0.0), (10.0, 0.0)), seen_crossings=seen)
        assert second == []


class TestMaskToSpans:
    """A class collecting tests for `mask_to_spans`."""

    t = np.arange(10.0)

    def test_two_runs(self) -> None:
        """Test extraction of two separate True runs."""
        mask = np.array([False, False, True, True, True, False, False, True, True, False])
        assert mask_to_spans(self.t, mask) == [(2.0, 4.0), (7.0, 8.0)]

    def test_runs_touching_edges(self) -> None:
        """Test runs that start at the first and end at the last element."""
        mask = np.array([True, True, False, False, False, False, False, False, True, True])
        assert mask_to_spans(self.t, mask) == [(0.0, 1.0), (8.0, 9.0)]

    def test_all_true(self) -> None:
        """Test a mask that is True everywhere."""
        assert mask_to_spans(self.t, np.ones(10, dtype=bool)) == [(0.0, 9.0)]

    def test_all_false(self) -> None:
        """Test a mask that is False everywhere."""
        assert mask_to_spans(self.t, np.zeros(10, dtype=bool)) == []

    def test_single_point_run(self) -> None:
        """Test that a run of length one yields a zero-width span."""
        mask = np.zeros(10, dtype=bool)
        mask[3] = True
        assert mask_to_spans(self.t, mask) == [(3.0, 3.0)]

    def test_shape_mismatch(self) -> None:
        """Test that differing shapes raise ValueError."""
        with pytest.raises(ValueError, match="must have the same shape"):
            mask_to_spans(self.t, np.ones(5, dtype=bool))

    def test_multi_dimensional_t(self) -> None:
        """Test that a two-dimensional t raises ValueError."""
        with pytest.raises(ValueError, match="must be one-dimensional"):
            mask_to_spans(np.zeros((2, 5)), np.ones((2, 5), dtype=bool))


class TestShadeSpans:
    """A class collecting tests for `shade_spans`."""

    t = np.arange(10.0)
    mask = np.array([False, False, True, True, True, False, False, True, True, False])

    def test_one_patch_per_run(self) -> None:
        """Test that each True run is shaded with one span of matching extent."""
        ax = Figure().add_subplot()
        result = shade_spans(ax, self.t, self.mask)
        n_runs = 2
        assert result is ax
        assert len(ax.patches) == n_runs
        spans: list[tuple[float, float]] = []
        for patch in ax.patches:
            assert isinstance(patch, Rectangle)
            bbox = patch.get_bbox()
            spans.append((bbox.x0, bbox.x1))
        assert spans == [(2.0, 4.0), (7.0, 8.0)]

    def test_color_and_alpha(self) -> None:
        """Test that the fill color and transparency are applied."""
        ax = Figure().add_subplot()
        shade_spans(ax, self.t, self.mask, color="C2", alpha=0.4)
        expected = mpl.colors.to_rgba("C2", alpha=0.4)
        assert ax.patches[0].get_facecolor() == pytest.approx(expected)

    def test_kwargs_passthrough(self) -> None:
        """Test that extra keyword arguments reach axvspan."""
        ax = Figure().add_subplot()
        zorder = -5
        shade_spans(ax, self.t, self.mask, zorder=zorder)
        assert ax.patches[0].get_zorder() == zorder

    def test_empty_mask_adds_nothing(self) -> None:
        """Test that an all-False mask draws no spans."""
        ax = Figure().add_subplot()
        shade_spans(ax, self.t, np.zeros(10, dtype=bool))
        assert len(ax.patches) == 0


class TestSetupAxes:
    """A class collecting tests for `setup_axes`."""

    def test_labels_title_and_limits(self) -> None:
        """Test that labels, title, and axis limits are applied."""
        ax = Figure().add_subplot()
        result = setup_axes(
            ax,
            xlabel="time [s]",
            ylabel="position [m]",
            title="hop",
            xlim=(0.0, 10.0),
            ylim=(-1.0, 1.0),
        )
        assert result is ax
        assert ax.get_xlabel() == "time [s]"
        assert ax.get_ylabel() == "position [m]"
        assert ax.get_title() == "hop"
        assert ax.get_xlim() == (0.0, 10.0)
        assert ax.get_ylim() == (-1.0, 1.0)

    def test_defaults_leave_axes_untouched(self) -> None:
        """Test that omitted options do not modify existing settings."""
        ax = Figure().add_subplot()
        ax.set_xlabel("keep me")
        ax.set_xlim(3.0, 4.0)
        setup_axes(ax, ylabel="new")
        assert ax.get_xlabel() == "keep me"
        assert ax.get_xlim() == (3.0, 4.0)
        assert ax.get_ylabel() == "new"

    @pytest.mark.parametrize(
        ("grid", "x_visible", "y_visible"),
        [
            (True, True, True),
            ("both", True, True),
            ("x", True, False),
            ("y", False, True),
            (False, False, False),
        ],
    )
    def test_grid(self, *, grid: bool | str, x_visible: bool, y_visible: bool) -> None:
        """Test grid visibility control per axis."""
        ax = Figure().add_subplot()
        setup_axes(ax, grid=grid)  # type: ignore[arg-type]
        assert ax.xaxis.get_gridlines()[0].get_visible() is x_visible
        assert ax.yaxis.get_gridlines()[0].get_visible() is y_visible

    def test_legend(self) -> None:
        """Test that a legend is created with the requested frame transparency."""
        ax = Figure().add_subplot()
        ax.plot([0.0, 1.0], [0.0, 1.0], label="line")
        setup_axes(ax, legend=True, legend_framealpha=0.5)
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_frame().get_alpha() == pytest.approx(0.5)

    def test_no_legend_by_default(self) -> None:
        """Test that no legend is created unless requested."""
        ax = Figure().add_subplot()
        ax.plot([0.0, 1.0], [0.0, 1.0], label="line")
        setup_axes(ax)
        assert ax.get_legend() is None


class TestAnnotateWithArrow:
    """A class collecting tests for `annotate_with_arrow`."""

    def test_positions(self) -> None:
        """Test that the arrow points at xy and the label sits at the offset in points."""
        ax = Figure().add_subplot()
        annotation = annotate_with_arrow(ax, r"$z$", (1.0, 2.0), (30.0, -6.0))
        assert annotation.get_text() == r"$z$"
        assert annotation.xy == (1.0, 2.0)
        assert annotation.xyann == (30.0, -6.0)
        assert annotation.anncoords == "offset points"

    def test_colors_default_to_shared_color(self) -> None:
        """Test that the arrow, box edge, and text share the color by default."""
        ax = Figure().add_subplot()
        annotation = annotate_with_arrow(ax, "z", (0.0, 0.0), (10.0, 10.0), color="C1")
        expected = mpl.colors.to_rgba("C1")
        assert mpl.colors.to_rgba(annotation.get_color()) == expected
        assert annotation.arrow_patch is not None
        assert annotation.arrow_patch.get_edgecolor() == expected
        bbox_patch = annotation.get_bbox_patch()
        assert bbox_patch is not None
        assert bbox_patch.get_edgecolor() == expected
        assert bbox_patch.get_facecolor() == mpl.colors.to_rgba("white")

    def test_separate_text_and_face_colors(self) -> None:
        """Test that the text and box background colors can differ from the shared color."""
        ax = Figure().add_subplot()
        annotation = annotate_with_arrow(
            ax,
            "z",
            (0.0, 0.0),
            (10.0, 10.0),
            color="C0",
            text_color="black",
            face_color="yellow",
        )
        assert mpl.colors.to_rgba(annotation.get_color()) == mpl.colors.to_rgba("black")
        bbox_patch = annotation.get_bbox_patch()
        assert bbox_patch is not None
        assert bbox_patch.get_facecolor() == mpl.colors.to_rgba("yellow")
        assert bbox_patch.get_edgecolor() == mpl.colors.to_rgba("C0")

    def test_line_width_and_pad(self) -> None:
        """Test that the line width and box margin are applied to box and arrow."""
        ax = Figure().add_subplot()
        annotation = annotate_with_arrow(ax, "z", (0.0, 0.0), (10.0, 10.0), lw=1.2, pad=0.05)
        bbox_patch = annotation.get_bbox_patch()
        assert bbox_patch is not None
        assert bbox_patch.get_linewidth() == pytest.approx(1.2)
        assert bbox_patch.get_boxstyle().pad == pytest.approx(0.05)  # type: ignore[attr-defined]
        assert annotation.arrow_patch is not None
        assert annotation.arrow_patch.get_linewidth() == pytest.approx(1.2)

    def test_fontsize_and_alignment_defaults(self) -> None:
        """Test the font size setting and the default text alignment."""
        ax = Figure().add_subplot()
        annotation = annotate_with_arrow(ax, "z", (0.0, 0.0), (10.0, 10.0), fontsize=12)
        assert annotation.get_fontsize() == pytest.approx(12)
        assert annotation.get_horizontalalignment() == "center"
        assert annotation.get_verticalalignment() == "bottom"

    def test_alignment_override(self) -> None:
        """Test that extra keyword arguments reach Axes.annotate."""
        ax = Figure().add_subplot()
        annotation = annotate_with_arrow(ax, "z", (0.0, 0.0), (10.0, 10.0), ha="left", va="top")
        assert annotation.get_horizontalalignment() == "left"
        assert annotation.get_verticalalignment() == "top"

    def test_relpos_and_shrink(self) -> None:
        """Test that the arrow tail position and shrink gaps are forwarded."""
        ax = Figure().add_subplot()
        annotation = annotate_with_arrow(
            ax,
            "z",
            (0.0, 0.0),
            (10.0, 10.0),
            relpos=(1.0, 0.5),
            shrink_a=0.2,
            shrink_b=1.5,
        )
        assert annotation.arrowprops is not None
        assert annotation.arrowprops["relpos"] == (1.0, 0.5)
        assert annotation.arrowprops["shrinkA"] == pytest.approx(0.2)
        assert annotation.arrowprops["shrinkB"] == pytest.approx(1.5)


class TestSaveFigure:
    """A class collecting tests for `save_figure`."""

    @staticmethod
    def make_figure() -> Figure:
        """Return a small figure with a single line plot."""
        fig = Figure()
        ax = fig.add_subplot()
        ax.plot([0.0, 1.0], [0.0, 1.0])
        return fig

    def test_save_multiple_extensions(self, tmp_path: Path) -> None:
        """Test saving a figure to multiple file formats."""
        fig = self.make_figure()
        paths = save_figure(fig, tmp_path, "myfig", ["png", ".pdf"])
        assert set(paths) == {tmp_path / "myfig.png", tmp_path / "myfig.pdf"}
        assert all(p.is_file() for p in paths)

    def test_no_extensions_saves_nothing(self, tmp_path: Path) -> None:
        """Test that no files are written when extensions is None."""
        fig = self.make_figure()
        assert save_figure(fig, tmp_path, "myfig", None) == []
        assert list(tmp_path.iterdir()) == []

    def test_none_output_directory_raises(self) -> None:
        """Test that a None output directory raises ValueError."""
        fig = self.make_figure()
        with pytest.raises(ValueError, match="'None' is not allowed"):
            save_figure(fig, None, "myfig", "png")  # type: ignore[arg-type]


def test_calculate_mean_err_se_divides_by_trial_count() -> None:
    """Test that standard error scales the deviation by the number of trials, not time points."""
    n_trials = 4
    rng = np.random.default_rng(seed=42)
    data_array = rng.random((n_trials, 10))
    mean, err1, err2 = calculate_mean_err(data_array, err_type="se")
    np.testing.assert_allclose(mean, np.mean(data_array, axis=0))
    np.testing.assert_allclose(err1, np.std(data_array, axis=0) / np.sqrt(n_trials))
    assert err2 is None


def test_fill_between_err_range_covers_data_extremes() -> None:
    """Test that the filled band for range errors spans the data minimum to maximum at each time."""
    t = np.array([1.0, 2.0, 3.0])
    y_arr = np.array([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0], [2.0, 10.0, 4.0]])
    ax = Figure().add_subplot()
    fill_between_err(ax, t, y_arr, "range", tlim=None, color=None, alpha=None)
    vertices = np.asarray(ax.collections[0].get_paths()[0].vertices)
    for i, x in enumerate(t):
        band_y = vertices[np.isclose(vertices[:, 0], x), 1]
        assert band_y.min() == pytest.approx(y_arr[:, i].min())
        assert band_y.max() == pytest.approx(y_arr[:, i].max())


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["./a/b/c/d.pdf", "./a/b/c/e.pdf"], Path("./a/b/c")),
        (["./a/b/c/d.pdf", "a/b/f/g.pdf", "./a/b/h/i.pdf"], Path("a/b")),
        (["a/b/c/d.pdf", "a/e/f/g.pdf", "./a/h/i.pdf"], Path("a")),
        (["a/b.pdf", "c/d.pdf", "e/f.pdf"], Path()),
        (["/a/b/c/d.pdf", "/a/b/f/g.pdf"], Path("/a/b")),
        (["/a/b.pdf", "/c/d.pdf", "/e/f.pdf"], Path("/")),
        ([Path.cwd() / "a/b.pdf", "a/c.pdf"], Path.cwd() / "a"),
        ([DATA_DIR_PATH / "test.csv"], DATA_DIR_PATH),
    ],
)
def test_extract_common_path(paths: list[str | Path], expected: Path) -> None:
    """Test common path extraction from multiple paths."""
    result = extract_common_path(*paths)
    assert result == expected


@pytest.mark.parametrize(
    ("xlim", "fallback", "fallback_xlim", "expected"),
    [
        (None, None, None, None),
        (None, (0.1, 0.2), None, None),
        (None, None, (0.3, 0.4), None),
        (None, (0.1, 0.2), (0.3, 0.4), None),
        ([], None, None, None),
        ([], (0.1, 0.2), None, (0.1, 0.2)),
        ([], None, (0.3, 0.4), (0.3, 0.4)),
        ([], (0.5, 0.6), (0.7, 0.8), (0.7, 0.8)),
        ([1.0], None, None, (-1.0, 1.0)),
        ([2.0], (0.1, 0.2), None, (-2.0, 2.0)),
        ([-3.0], None, (0.3, 0.4), (-3.0, 3.0)),
        ([-4.0], (0.5, 0.6), (0.7, 0.8), (-4.0, 4.0)),
        ((1.0, 2.0), None, None, (1.0, 2.0)),
        ((4.0, 3.0), (0.1, 0.2), None, (3.0, 4.0)),
        ((5.0, -6.0), None, (0.3, 0.4), (-6.0, 5.0)),
        ([-7.0, 8.0], (0.5, 0.6), (0.7, 0.8), (-7.0, 8.0)),
        ((1.0, 2.0, 3.0), None, None, (1.0, 3.0)),
        ((5.0, 4.0, 7.0, 6.0), (0.1, 0.2), None, (4.0, 7.0)),
        ([7.0, -8.0, 9.0], None, (0.3, 0.4), (-8.0, 9.0)),
        (range(10), (0.5, 0.6), (0.7, 0.8), (0, 9)),
    ],
)
def test_get_limits_xlim(
    xlim: Sequence[float] | None,
    fallback: tuple[float, float] | None,
    fallback_xlim: tuple[float, float] | None,
    expected: tuple[float, float] | None,
) -> None:
    """Test get_limits function with x-axis limits only."""
    fixed_xlim = get_limits(xlim, fallback=fallback, fallback_xlim=fallback_xlim)
    assert fixed_xlim == expected


@pytest.mark.parametrize(
    ("xlim", "ylim", "fallback", "fallback_xlim", "fallback_ylim", "expected"),
    [
        (None, None, None, None, None, (None, None)),
        (None, [], (0.1, 0.2), None, None, (None, (0.1, 0.2))),
        (None, [], None, None, (0.3, 0.4), (None, (0.3, 0.4))),
        (None, [], (0.1, 0.2), None, (0.3, 0.4), (None, (0.3, 0.4))),
        (None, (1.0, 2.0), (0.1, 0.2), None, None, (None, (1.0, 2.0))),
        (None, (1.0, -2.0), None, None, (0.3, 0.4), (None, (-2.0, 1.0))),
        (None, [-1.0, -2.0], (0.1, 0.2), None, (0.3, 0.4), (None, (-2.0, -1.0))),
        (None, (1.0, 2.0, 3.0), (0.1, 0.2), None, None, (None, (1.0, 3.0))),
        (None, [1.0, -2.0, 3.0], None, None, (0.3, 0.4), (None, (-2.0, 3.0))),
        (None, (-1.0, 2.0, -3.0), (0.1, 0.2), None, (0.3, 0.4), (None, (-3.0, 2.0))),
        ([], [], (0.1, 0.2), None, None, ((0.1, 0.2), (0.1, 0.2))),
        ([], [], None, (0.3, 0.4), None, ((0.3, 0.4), None)),
        ([], [], None, None, (0.5, 0.6), (None, (0.5, 0.6))),
        ([], [], (0.1, 0.2), (0.3, 0.4), None, ((0.3, 0.4), (0.1, 0.2))),
        ([], [], (0.1, 0.2), None, (0.3, 0.4), ((0.1, 0.2), (0.3, 0.4))),
        ([], [], None, (0.1, 0.2), (0.3, 0.4), ((0.1, 0.2), (0.3, 0.4))),
        ([], [], (0.1, 0.2), (0.3, 0.4), (0.5, 0.6), ((0.3, 0.4), (0.5, 0.6))),
        ([1.0], [2.0], (0.1, 0.2), None, None, ((-1.0, 1.0), (-2.0, 2.0))),
        ([3.0, -3.0], [4.0, -4.0], None, (0.3, 0.4), None, ((-3.0, 3.0), (-4.0, 4.0))),
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), None, None, (0.5, 0.6), ((1.0, 3.0), (4.0, 6.0))),
        ([1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], (0.1, 0.2), (0.3, 0.4), None, ((-2.0, 3.0), (-6.0, 5.0))),
        ([1.0, -2.0], [3.0], (0.1, 0.2), None, (0.3, 0.4), ((-2.0, 1.0), (-3.0, 3.0))),
        ((-1.0, 2.0, 3.0), [4.0], None, (0.1, 0.2), (0.3, 0.4), ((-1.0, 3.0), (-4.0, 4.0))),
        (range(5), range(10), (0.1, 0.2), (0.3, 0.4), (0.5, 0.6), ((0, 4), (0, 9))),
    ],
)
def test_get_limits(
    xlim: Sequence[float] | None,
    ylim: Sequence[float] | None,
    fallback: tuple[float, float] | None,
    fallback_xlim: tuple[float, float] | None,
    fallback_ylim: tuple[float, float] | None,
    expected: tuple[tuple[float, float] | None, tuple[float, float] | None],
) -> None:
    """Test get_limits function with both x and y axis limits."""
    fixed_xlim, fixed_ylim = get_limits(
        xlim,
        ylim,
        fallback=fallback,
        fallback_xlim=fallback_xlim,
        fallback_ylim=fallback_ylim,
    )
    assert fixed_xlim == expected[0]
    assert fixed_ylim == expected[1]


# Local Variables:
# jinx-local-words: "axb basename cmap csv cxx dat ddof dir err figsize jpg linewidth myfig noqa parametrized pdf plotutil png px pyplotutil pytest rcParams str tlim usetex viridis xdata xlim ylim" # noqa: E501
# End:
