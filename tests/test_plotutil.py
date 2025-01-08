# ruff: noqa: S101
"""Unit tests for plotting utility functions in pyplotutil.plotutil module.

This test suite covers:
- Filename compatibility conversion for both string and Path objects
- Figure path generation with various configurations:
    * Single and multiple extensions
    * Directory separation by main module
    * Directory separation by extension
    * Duplicate extension handling
- Common path extraction functionality

The tests use parametrized fixtures to verify multiple input scenarios
and edge cases for each function.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from pyplotutil.plotutil import compatible_filename, extract_common_path, make_figure_paths
from tests.test_datautil import DATA_DIR_PATH

if TYPE_CHECKING:
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
    """Test string filename compatibility conversion.

    Parameters
    ----------
    filename : str
        Input filename to test
    expected : str
        Expected converted filename

    """
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
    """Test Path object filename compatibility conversion.

    Parameters
    ----------
    filename : str
        Input filename to convert to Path
    expected : str
        Expected converted filename

    """
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
    """Test figure path generation with single extension.

    Parameters
    ----------
    output_directory : FilePath
        Directory for output files
    basename : str
        Base name for the figure
    extension : str
        Single file extension
    expected : str
        Expected figure path

    """
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
    """Test figure path generation with multiple extensions.

    Parameters
    ----------
    output_directory : FilePath
        Directory for output files
    basename : str
        Base name for the figure
    extensions : list[str]
        List of file extensions
    expected : list[str]
        Expected figure paths

    """
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
    """Test duplicate extension removal in figure path generation.

    Parameters
    ----------
    output_directory : FilePath
        Directory for output files
    basename : str
        Base name for the figure
    extensions : list[str]
        List of file extensions with duplicates
    expected : list[str]
        Expected deduplicated figure paths

    """
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
    output_directory: FilePath,
    basename: str,
    extensions: list[str],
    main_module: bool | str,
    expected: list[str],
) -> None:
    """Test figure path generation with main module directory separation.

    Parameters
    ----------
    output_directory : FilePath
        Directory for output files
    basename : str
        Base name for the figure
    extensions : list[str]
        List of file extensions
    main_module : bool or str
        Main module separation flag or name
    expected : list[str]
        Expected figure paths with module separation

    """
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
    """Test figure path generation with extension directory separation.

    Parameters
    ----------
    output_directory : FilePath
        Directory for output files
    basename : str
        Base name for the figure
    extensions : list[str]
        List of file extensions
    expected : list[str]
        Expected figure paths with extension separation

    """
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
    """Test comprehensive figure path generation with all options.

    Parameters
    ----------
    output_directory : FilePath
        Directory for output files
    basename : str
        Base name for the figure
    extensions : list[str]
        List of file extensions
    main_module : bool or str
        Main module separation flag or name
    separate_dir_by_ext : bool
        Flag for extension directory separation
    expected : list[str]
        Expected figure paths

    """
    figure_paths = make_figure_paths(
        output_directory,
        basename,
        extensions,
        separate_dir_by_main_module=main_module,
        separate_dir_by_ext=separate_dir_by_ext,
    )
    assert set(figure_paths) == {Path(e) for e in expected}


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
    """Test common path extraction from multiple paths.

    Parameters
    ----------
    paths : list[str | Path]
        List of input paths
    expected : Path
        Expected common path

    """
    result = extract_common_path(*paths)
    assert result == expected
