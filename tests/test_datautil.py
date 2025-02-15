# ruff: noqa: S101
"""Unit tests for the datautil module.

This test suite verifies the functionality of the `Data` and `TaggedData` classes, which facilitate
the management and manipulation of tabular data in pandas DataFrames. The tests cover class
initialization, attribute access, data indexing, and various utility methods to ensure robust and
consistent behavior in data handling and processing.

"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy.testing as nt
import pandas as pd
import pandas.testing as pt
import pytest

from pyplotutil.datautil import Data, DataSourceType, TaggedData

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from pandas.io.parsers.readers import UsecolsArgType

DATA_DIR_PATH = Path(__file__).parent / "data"
TEST_CSV_FILE_PATH = DATA_DIR_PATH / "test.csv"
TEST_TAGGED_DATA_CSV_FILE_PATH = DATA_DIR_PATH / "test_tagged_data.csv"

TEST_TEXT = """\
a,b,c,d,e
1,0.01,10.0,3.5,100
2,0.02,20.0,7.5,200
3,0.03,30.0,9.5,300
4,0.04,40.0,11.5,400
"""

TEST_NO_HEADER_TEXT = """\
1,0.01,10.0,3.5,100
2,0.02,20.0,7.5,200
3,0.03,30.0,9.5,300
4,0.04,40.0,11.5,400
"""

TEST_COMMENT_HEADER_TEXT = """\
# a,b,c,d,e
1,0.01,10.0,3.5,100
2,0.02,20.0,7.5,200
3,0.03,30.0,9.5,300
4,0.04,40.0,11.5,400
"""

TEST_TAGGED_DATA_TEXT = """\
tag,a,b,c,d,e
tag01,0,1,2,3,4
tag01,5,6,7,8,9
tag01,10,11,12,13,14
tag01,15,16,17,18,19
tag01,20,21,22,23,24
tag01,25,26,27,28,29
tag02,10,11,12,13,14
tag02,15,16,17,18,19
tag02,110,111,112,113,114
tag02,115,116,117,118,119
tag02,120,121,122,123,124
tag02,125,126,127,128,129
tag03,20,21,22,23,24
tag03,25,26,27,28,29
tag03,210,211,212,213,214
tag03,215,216,217,218,219
tag03,220,221,222,223,224
tag03,225,226,227,228,229
"""


@pytest.fixture(scope="session")
def expected_dataframe() -> pd.DataFrame:
    """Return a pandas DataFrame object loaded from test.csv."""
    return pd.read_csv(TEST_CSV_FILE_PATH)


@pytest.fixture(scope="session")
def default_data() -> Data:
    """Return a default `Data` object."""
    return Data(TEST_CSV_FILE_PATH)


@pytest.fixture(scope="session")
def tagged_dataframe() -> pd.DataFrame:
    """Return a pandas DataFrame object loaded from test_tagged_data.csv."""
    return pd.read_csv(TEST_TAGGED_DATA_CSV_FILE_PATH)


@pytest.fixture(scope="session")
def default_tagged_data() -> TaggedData:
    """Return a default `Data` object."""
    return TaggedData(TEST_TAGGED_DATA_CSV_FILE_PATH)


def test_data_init_with_dataframe(expected_dataframe: pd.DataFrame) -> None:
    """Test the initialization of a `Data` object from a pandas DataFrame."""
    data = Data(expected_dataframe)
    assert data.dataframe is expected_dataframe
    pt.assert_frame_equal(data.dataframe, expected_dataframe)


def test_data_init_with_series() -> None:
    """Test the initialization of a `Data` object from a pandas DataFrame."""
    values = [1, 2, 3, 4]
    name = "a"
    series = pd.Series(values, name=name)
    expected_dataframe = pd.DataFrame(values, columns=pd.Series([name]))
    data = Data(series)
    pt.assert_frame_equal(data.dataframe, expected_dataframe)


def test_data_init_with_series_no_name() -> None:
    """Test the initialization of a `Data` object from a pandas DataFrame."""
    values = [1, 2, 3, 4]
    series = pd.Series(values)
    expected_dataframe = pd.DataFrame(values)
    data = Data(series)
    pt.assert_frame_equal(data.dataframe, expected_dataframe)


def test_data_init_with_string_buffer(expected_dataframe: pd.DataFrame) -> None:
    """Test the initialization of a `Data` object from a `StringIO` object."""
    data = Data(StringIO(TEST_TEXT))
    pt.assert_frame_equal(data.dataframe, expected_dataframe)


def test_data_init_with_path_str(expected_dataframe: pd.DataFrame) -> None:
    """Test the initialization of a `Data` object from a file path string."""
    data = Data(str(TEST_CSV_FILE_PATH))
    assert data.datapath == TEST_CSV_FILE_PATH
    assert data.datadir == DATA_DIR_PATH
    pt.assert_frame_equal(data.dataframe, expected_dataframe)


def test_data_init_with_path_object(expected_dataframe: pd.DataFrame) -> None:
    """Test the initialization of a `Data` object from a file path object."""
    data = Data(Path(TEST_CSV_FILE_PATH))
    assert data.datapath == TEST_CSV_FILE_PATH
    assert data.datadir == DATA_DIR_PATH
    pt.assert_frame_equal(data.dataframe, expected_dataframe)


@pytest.mark.parametrize(
    ("filepath", "sep"),
    [
        (DATA_DIR_PATH / "test.csv", ","),
        (DATA_DIR_PATH / "test_spaces.txt", " "),
    ],
)
def test_data_init_with_sep(expected_dataframe: pd.DataFrame, filepath: Path, sep: str) -> None:
    """Test the initialization of a `Data` object from a file with `sep` parameter."""
    data = Data(filepath, sep=sep)
    assert data.datapath == filepath
    assert data.datadir == DATA_DIR_PATH
    pt.assert_frame_equal(data.dataframe, expected_dataframe)


@pytest.mark.parametrize(
    ("data_source", "header", "names"),
    [
        (TEST_CSV_FILE_PATH, "infer", None),
        (TEST_CSV_FILE_PATH, 0, None),
        (DATA_DIR_PATH / "test_no_header.csv", None, ["a", "b", "c", "d", "e"]),
        (StringIO(TEST_TEXT), "infer", None),
        (StringIO(TEST_TEXT), 0, None),
        (StringIO(TEST_NO_HEADER_TEXT), None, ["a", "b", "c", "d", "e"]),
    ],
)
def test_data_init_with_header_and_names(
    expected_dataframe: pd.DataFrame,
    data_source: DataSourceType,
    header: Literal["infer"] | int | None,
    names: Sequence[Hashable] | None,
) -> None:
    """Test the initialization of a `Data` object from a file with `header` and `names` parameters."""
    data = Data(data_source, header=header, names=names)
    pt.assert_frame_equal(data.dataframe, expected_dataframe)


@pytest.mark.parametrize("data_source", [TEST_CSV_FILE_PATH, StringIO(TEST_TEXT)])
@pytest.mark.parametrize(
    "usecols",
    [
        [0, 1, 2],
        range(3),
        [1, 2, 0],
        ["a", "b", "c"],
    ],
)
def test_data_init_with_usecols(
    expected_dataframe: pd.DataFrame,
    data_source: DataSourceType,
    usecols: UsecolsArgType,
) -> None:
    """Test the initialization of a `Data` object from a file with `usecols` parameter."""
    data = Data(data_source, usecols=usecols)
    expected = expected_dataframe.iloc[:, [0, 1, 2]]
    pt.assert_frame_equal(data.dataframe, expected)


@pytest.mark.parametrize("data_source", [TEST_CSV_FILE_PATH, StringIO(TEST_TEXT)])
@pytest.mark.parametrize("nrows", [2, 3])
def test_data_init_with_nrows(
    expected_dataframe: pd.DataFrame,
    data_source: DataSourceType,
    nrows: int,
) -> None:
    """Test the initialization of a `Data` object from a file with `nrows` parameter."""
    data = Data(data_source, nrows=nrows)
    expected = expected_dataframe[:nrows]
    assert len(data) == nrows
    pt.assert_frame_equal(data.dataframe, expected)


@pytest.mark.parametrize(
    ("data_source", "comment"),
    [
        (DATA_DIR_PATH / "test_comment_header.csv", "#"),
        (StringIO(TEST_COMMENT_HEADER_TEXT), "#"),
    ],
)
def test_data_read_commented_header(
    expected_dataframe: pd.DataFrame,
    data_source: DataSourceType,
    comment: str,
) -> None:
    """Test the initialization of a `Data` object from a file with commented header."""
    data = Data(data_source, comment=comment)
    pt.assert_frame_equal(data.dataframe, expected_dataframe)


def test_data_read_partially_commented_data(expected_dataframe: pd.DataFrame) -> None:
    """Test the initialization of a `Data` object from a file with commented header."""
    data_source = DATA_DIR_PATH / "test_comment_partial_data.csv"
    data = Data(data_source, comment="#")
    expected = expected_dataframe.iloc[[0, 3]].reset_index(drop=True)
    pt.assert_frame_equal(data.dataframe, expected)


@pytest.mark.parametrize(
    ("data_source", "expected"),
    [
        (TEST_CSV_FILE_PATH, True),
        (StringIO(TEST_TEXT), False),
        (pd.read_csv(TEST_CSV_FILE_PATH), False),
    ],
)
def test_is_loaded_from_file(data_source: DataSourceType, *, expected: bool) -> None:
    """Test if a `Data` object is loaded from a file."""
    data = Data(data_source)
    assert data.is_loaded_from_file() is expected


@pytest.mark.parametrize("data_source", [StringIO(TEST_TEXT), pd.read_csv(TEST_CSV_FILE_PATH)])
def test_datapath_error(data_source: DataSourceType) -> None:
    """Test if an exception is raised when access to data path after initialized without a file."""
    data = Data(data_source)
    msg = "Data object may not be loaded from a file."
    with pytest.raises(AttributeError, match=msg):
        _ = data.datapath


@pytest.mark.parametrize("data_source", [StringIO(TEST_TEXT), pd.read_csv(TEST_CSV_FILE_PATH)])
def test_datadir_error(data_source: DataSourceType) -> None:
    """Test if an exception is raised when access to data path after initialized without a file."""
    data = Data(data_source)
    msg = "Data object may not be loaded from a file."
    with pytest.raises(AttributeError, match=msg):
        _ = data.datadir


@pytest.fixture
def toy_dataframe() -> pd.DataFrame:
    """Return a toy DataFrame."""
    raw_data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    columns = pd.Series(["a", "b", "c"])
    return pd.DataFrame(raw_data, columns=columns)


def test_data_getitem(toy_dataframe: pd.DataFrame) -> None:
    """Test column access via indexing on `Data` objects."""
    data = Data(toy_dataframe)

    pt.assert_series_equal(data["a"], toy_dataframe.a)
    pt.assert_series_equal(data["b"], toy_dataframe.b)
    pt.assert_series_equal(data["c"], toy_dataframe.c)


def test_data_getitem_multiple(toy_dataframe: pd.DataFrame) -> None:
    """Test multiple column access via indexing on `Data` objects."""
    data = Data(toy_dataframe)

    pt.assert_frame_equal(data[["a", "b"]], toy_dataframe.loc[:, ["a", "b"]])
    pt.assert_frame_equal(data[["a", "b"]], toy_dataframe[["a", "b"]])


def test_data_getitem_no_header() -> None:
    """Test column access in DataFrames without a header."""
    toy_dataframe_no_header = pd.DataFrame([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    data = Data(toy_dataframe_no_header)

    pt.assert_series_equal(data[0], toy_dataframe_no_header[0])
    pt.assert_series_equal(data[1], toy_dataframe_no_header[1])
    pt.assert_series_equal(data[2], toy_dataframe_no_header[2])


def test_data_getitem_no_header_multiple() -> None:
    """Test multiple column access in DataFrames without a header."""
    toy_dataframe_no_header = pd.DataFrame([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    data = Data(toy_dataframe_no_header)

    pt.assert_frame_equal(data[[0, 1]], toy_dataframe_no_header.loc[:, [0, 1]])
    pt.assert_frame_equal(data[[0, 1]], toy_dataframe_no_header[[0, 1]])


def test_data_len(toy_dataframe: pd.DataFrame) -> None:
    """Test length access via the `__len__` method."""
    data = Data(toy_dataframe)

    assert len(data) == len(toy_dataframe)


def test_data_getattr(toy_dataframe: pd.DataFrame) -> None:
    """Test attribute-style access to various DataFrame attributes."""
    data = Data(toy_dataframe)

    pt.assert_index_equal(data.columns, pd.Index(["a", "b", "c"]))
    assert data.shape == (3, 3)
    assert data.to_csv() == ",a,b,c\n0,0,1,2\n1,3,4,5\n2,6,7,8\n"
    expected = 5
    assert data.iat[1, 2] == expected  # noqa: PD009
    assert data.iloc[1, 2] == expected
    expected = 6
    assert data.at[2, "a"] == expected  # noqa: PD008
    assert data.loc[2, "a"] == expected


def test_data_attributes(toy_dataframe: pd.DataFrame) -> None:
    """Test direct attribute access for columns."""
    data = Data(toy_dataframe)

    pt.assert_series_equal(data.a, toy_dataframe.a)
    pt.assert_series_equal(data.b, toy_dataframe.b)
    pt.assert_series_equal(data.c, toy_dataframe.c)


def test_data_iterator(toy_dataframe: pd.DataFrame) -> None:
    """Test iteration over Data object."""
    data = Data(toy_dataframe)
    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    for values, e in zip(data, expected, strict=True):
        nt.assert_array_equal(values, e)


@pytest.mark.parametrize("row_index", [1, 2])
def test_data_split(default_data: Data, row_index: int) -> None:
    """Test splitting a Data object into two objects by row index."""
    n_rows = len(default_data)
    n_cols = len(default_data.columns)
    train_data, test_data = default_data.split_by_row(row_index)
    assert train_data.shape == (row_index, n_cols)
    pt.assert_index_equal(train_data.index, pd.RangeIndex(stop=row_index))
    assert test_data.shape == (n_rows - row_index, n_cols)
    pt.assert_index_equal(test_data.index, pd.RangeIndex(stop=n_rows - row_index))


@pytest.mark.parametrize(
    ("col", "expected"),
    [
        ("a", 1),
        ("d", 3.5),
    ],
)
def test_data_min(default_data: Data, col: str, expected: float) -> None:
    """Test minimum value retrieval from specified columns."""
    assert default_data[col].min() == expected


@pytest.mark.parametrize(
    ("cols", "expected"),
    [
        (["a", "b", "c"], [1, 0.01, 10.0]),
        (["b", "d", "c", "e"], [0.01, 3.5, 10.0, 100]),
    ],
)
def test_data_min_list(default_data: Data, cols: list[str], expected: list[float]) -> None:
    """Test minimum value retrieval from multiple columns."""
    pt.assert_series_equal(default_data[cols].min(), pd.Series(expected, index=cols))


@pytest.mark.parametrize(
    ("col", "expected"),
    [
        ("b", 0.04),
        ("c", 40.0),
    ],
)
def test_data_max(default_data: Data, col: str, expected: float) -> None:
    """Test maximum value retrieval from specified columns."""
    assert default_data[col].max() == expected


@pytest.mark.parametrize(
    ("cols", "expected"),
    [
        (["a", "b", "c"], [4, 0.04, 40.0]),
        (["b", "d", "c", "e"], [0.04, 11.5, 40.0, 400]),
    ],
)
def test_data_max_list(default_data: Data, cols: list[str], expected: list[float]) -> None:
    """Test maximum value retrieval from multiple columns."""
    pt.assert_series_equal(default_data[cols].max(), pd.Series(expected, index=cols))


@pytest.mark.parametrize(
    ("col", "expected"),
    [
        ("b", 0.01),
    ],
)
def test_data_param(default_data: Data, col: str, expected: float) -> None:
    """Test parameter retrieval for a specified column."""
    assert default_data.param(col) == expected


@pytest.mark.parametrize(
    ("cols", "expected"),
    [
        (["c", "e"], [10.0, 100]),
    ],
)
def test_data_param_list(default_data: Data, cols: list[str], expected: list[float]) -> None:
    """Test parameter retrieval from multiple columns."""
    pt.assert_series_equal(default_data.param(cols), pd.Series(expected, index=cols), check_names=False)


def test_tagged_data_init_with_dataframe(tagged_dataframe: pd.DataFrame) -> None:
    """Test initialization of `TaggedData` from a pandas DataFrame."""
    tagged_data = TaggedData(tagged_dataframe)
    groups = tagged_dataframe.groupby("tag")

    pt.assert_frame_equal(tagged_data.dataframe, tagged_dataframe)
    pt.assert_frame_equal(
        tagged_data.datadict["tag01"].dataframe,
        groups.get_group("tag01").reset_index(drop=True),
    )
    pt.assert_frame_equal(
        tagged_data.datadict["tag02"].dataframe,
        groups.get_group("tag02").reset_index(drop=True),
    )
    pt.assert_frame_equal(
        tagged_data.datadict["tag03"].dataframe,
        groups.get_group("tag03").reset_index(drop=True),
    )


def test_tagged_data_init_with_string_buffer(tagged_dataframe: pd.DataFrame) -> None:
    """Test initialization of `TaggedData` from a `StringIO` object."""
    tagged_data = TaggedData(StringIO(TEST_TAGGED_DATA_TEXT))

    pt.assert_frame_equal(tagged_data.dataframe, tagged_dataframe)
    groups = tagged_dataframe.groupby("tag")
    datadict = tagged_data.datadict
    pt.assert_frame_equal(
        datadict["tag01"].dataframe,
        groups.get_group("tag01").reset_index(drop=True),
    )
    pt.assert_frame_equal(
        datadict["tag02"].dataframe,
        groups.get_group("tag02").reset_index(drop=True),
    )
    pt.assert_frame_equal(
        datadict["tag03"].dataframe,
        groups.get_group("tag03").reset_index(drop=True),
    )


def test_tagged_data_init_with_path_str(tagged_dataframe: pd.DataFrame) -> None:
    """Test initialization of `TaggedData` from a file path."""
    tagged_data = TaggedData(str(TEST_TAGGED_DATA_CSV_FILE_PATH))

    assert tagged_data.datapath == TEST_TAGGED_DATA_CSV_FILE_PATH
    assert tagged_data.datadir == DATA_DIR_PATH
    pt.assert_frame_equal(tagged_data.dataframe, tagged_dataframe)
    groups = tagged_dataframe.groupby("tag")
    datadict = tagged_data.datadict
    pt.assert_frame_equal(
        datadict["tag01"].dataframe,
        groups.get_group("tag01").reset_index(drop=True),
    )
    pt.assert_frame_equal(
        datadict["tag02"].dataframe,
        groups.get_group("tag02").reset_index(drop=True),
    )
    pt.assert_frame_equal(
        datadict["tag03"].dataframe,
        groups.get_group("tag03").reset_index(drop=True),
    )


def test_tagged_data_init_with_path_object(tagged_dataframe: pd.DataFrame) -> None:
    """Test initialization of `TaggedData` from a file path."""
    tagged_data = TaggedData(Path(TEST_TAGGED_DATA_CSV_FILE_PATH))

    assert tagged_data.datapath == TEST_TAGGED_DATA_CSV_FILE_PATH
    assert tagged_data.datadir == DATA_DIR_PATH
    pt.assert_frame_equal(tagged_data.dataframe, tagged_dataframe)
    groups = tagged_dataframe.groupby("tag")
    datadict = tagged_data.datadict
    pt.assert_frame_equal(
        datadict["tag01"].dataframe,
        groups.get_group("tag01").reset_index(drop=True),
    )
    pt.assert_frame_equal(
        datadict["tag02"].dataframe,
        groups.get_group("tag02").reset_index(drop=True),
    )
    pt.assert_frame_equal(
        datadict["tag03"].dataframe,
        groups.get_group("tag03").reset_index(drop=True),
    )


def test_tagged_data_init_with_custom_tag() -> None:
    """Test tagged data grouping with a custom tag column."""
    path = DATA_DIR_PATH / "test_tagged_data_label.csv"
    tagged_dataframe = pd.read_csv(path)

    tagged_data = TaggedData(path, tag="label")

    assert tagged_data.datapath == path
    assert tagged_data.datadir == DATA_DIR_PATH
    pt.assert_frame_equal(tagged_data.dataframe, tagged_dataframe)
    groups = tagged_dataframe.groupby("label")
    pt.assert_frame_equal(
        tagged_data.datadict["label01"].dataframe,
        groups.get_group("label01").reset_index(drop=True),
    )
    pt.assert_frame_equal(
        tagged_data.datadict["label02"].dataframe,
        groups.get_group("label02").reset_index(drop=True),
    )
    pt.assert_frame_equal(
        tagged_data.datadict["label03"].dataframe,
        groups.get_group("label03").reset_index(drop=True),
    )


def test_tagged_data_no_tag() -> None:
    """Test `TaggedData` initialization without a tag column."""
    tagged_dataframe = pd.read_csv(TEST_CSV_FILE_PATH)

    tagged_data = TaggedData(TEST_CSV_FILE_PATH)
    pt.assert_frame_equal(tagged_data.dataframe, tagged_dataframe)
    assert len(tagged_data.datadict) == 1
    pt.assert_frame_equal(tagged_data.datadict["unknown"].dataframe, tagged_dataframe)


def test_tagged_data_iterator(default_tagged_data: TaggedData, tagged_dataframe: pd.DataFrame) -> None:
    """Test iteration over `TaggedData` groups."""
    groups = tagged_dataframe.groupby("tag")
    for i, data in enumerate(default_tagged_data):
        pt.assert_frame_equal(
            data.dataframe,
            groups.get_group(f"tag{i+1:02d}").reset_index(drop=True),
        )


def test_tagged_data_datadict(default_tagged_data: TaggedData) -> None:
    """Test access to the `datadict` property in `TaggedData`."""
    assert isinstance(default_tagged_data.datadict, dict)
    assert list(default_tagged_data.datadict.keys()) == ["tag01", "tag02", "tag03"]


def test_tagged_data_tags(default_tagged_data: TaggedData) -> None:
    """Test retrieval of unique tags in `TaggedData`."""
    assert list(default_tagged_data.tags()) == ["tag01", "tag02", "tag03"]


def test_tagged_data_items(default_tagged_data: TaggedData, tagged_dataframe: pd.DataFrame) -> None:
    """Test the `items` method to retrieve data groups and associated tags."""
    groups = tagged_dataframe.groupby("tag")
    assert len(default_tagged_data.items()) == len(groups)
    for i, (tag, data) in enumerate(default_tagged_data.items()):
        assert tag == f"tag{i+1:02d}"
        pt.assert_frame_equal(
            data.dataframe,
            groups.get_group(tag).reset_index(drop=True),
        )


def test_tagged_data_get(default_tagged_data: TaggedData, tagged_dataframe: pd.DataFrame) -> None:
    """Test retrieval of a data group by tag."""
    groups = tagged_dataframe.groupby("tag")
    for tag in ["tag01", "tag02", "tag03"]:
        data = default_tagged_data.get(tag)
        pt.assert_frame_equal(
            data.dataframe,
            groups.get_group(tag).reset_index(drop=True),
        )


def test_tagged_data_get_default_data(
    default_tagged_data: TaggedData,
    tagged_dataframe: pd.DataFrame,
    default_data: Data,
) -> None:
    """Test retrieval of a data group by tag with default data object."""
    groups = tagged_dataframe.groupby("tag")
    for tag in ["tag01", "tag02", "tag03"]:
        data = default_tagged_data.get(tag, default=default_data)
        pt.assert_frame_equal(
            data.dataframe,
            groups.get_group(tag).reset_index(drop=True),
        )
    data_non_exist_tag = default_tagged_data.get("non_exist_tag", default=default_data)
    pt.assert_frame_equal(data_non_exist_tag.dataframe, default_data.dataframe)


def test_tagged_data_get_default_none(default_tagged_data: TaggedData, tagged_dataframe: pd.DataFrame) -> None:
    """Test retrieval of a data group by tag with None object."""
    groups = tagged_dataframe.groupby("tag")
    for tag in ["tag01", "tag02", "tag03"]:
        data = default_tagged_data.get(tag, default=None)
        if data is not None:
            pt.assert_frame_equal(
                data.dataframe,
                groups.get_group(tag).reset_index(drop=True),
            )
    data_non_exist_tag = default_tagged_data.get("non_exist_tag", default=None)
    assert data_non_exist_tag is None


@pytest.mark.parametrize(
    ("tag", "col", "expected"),
    [
        ("tag01", "a", 0),
        ("tag02", "b", 11),
        ("tag03", "c", 22),
    ],
)
def test_tagged_data_param(default_tagged_data: TaggedData, tag: str, col: str, expected: float) -> None:
    """Test parameter retrieval from a tagged group."""
    assert default_tagged_data.param(tag, col) == expected


@pytest.mark.parametrize(
    ("tag", "cols", "expected"),
    [
        ("tag01", ["a", "b", "c"], [0, 1, 2]),
        ("tag02", ["b", "c", "d"], [11, 12, 13]),
        ("tag03", ["c", "d", "e"], [22, 23, 24]),
    ],
)
def test_tagged_data_param_list(
    default_tagged_data: TaggedData,
    tag: str,
    cols: list[str],
    expected: list[float],
) -> None:
    """Test parameter retrieval for multiple columns from a tagged group."""
    pt.assert_series_equal(default_tagged_data.param(tag, cols), pd.Series(expected, index=cols), check_names=False)


def test_tagged_data_param_list_unpack(default_tagged_data: TaggedData) -> None:
    """Test unpacking multiple column parameters from a tagged group."""
    a, b, c = default_tagged_data.param("tag01", ["a", "b", "c"])
    expected_values = (0, 1, 2)
    assert a == expected_values[0]
    assert b == expected_values[1]
    assert c == expected_values[2]


# Local Variables:
# jinx-local-words: "StringIO cls csv datadict datautil filepath len noqa nrows sep txt usecols"
# End:
