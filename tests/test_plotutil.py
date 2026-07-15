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

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from pyplotutil.plotutil import (
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
from tests.test_datautil import DATA_DIR_PATH

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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

    def test_ci_not_implemented(self) -> None:
        """Test that confidence intervals are not implemented yet."""
        with pytest.raises(NotImplementedError):
            calculate_mean_err(self.data_array, err_type="ci")

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
