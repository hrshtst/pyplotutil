# ruff: noqa: ANN001,ANN003,PLR2004,T201,PGH003,PD901,PD008,PD009
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import pandas._testing as pt
import pytest

from pyplotutil.datautil import Data, TaggedData

csv_dir_path = Path(__file__).parent / "data"

test_data = """\
a,b,c,d,e
1,0.01,10.0,3.5,100
2,0.02,20.0,7.5,200
3,0.03,30.0,9.5,300
4,0.04,40.0,11.5,400
"""

test_tagged_data = """\
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


@pytest.mark.parametrize("cls", [str, Path])
def test_data_init_path(cls) -> None:
    csv_path = csv_dir_path / "test.csv"
    path = cls(csv_path)
    expected_df = pd.read_csv(csv_path)

    data = Data(path)

    assert data.datapath == Path(csv_path)
    assert data.datadir == Path(csv_dir_path)
    pt.assert_frame_equal(data.dataframe, expected_df)


def test_data_init_StringIO() -> None:
    csv_path = csv_dir_path / "test.csv"
    expected_df = pd.read_csv(csv_path)

    data = Data(StringIO(test_data))

    assert data.datapath is None
    assert data.datadir is None
    pt.assert_frame_equal(data.dataframe, expected_df)


def test_data_init_DataFrame() -> None:
    csv_path = csv_dir_path / "test.csv"
    expected_df = pd.read_csv(csv_path)

    if isinstance(expected_df, pd.DataFrame):
        data = Data(expected_df)

        assert data.datapath is None
        assert data.datadir is None
        pt.assert_frame_equal(data.dataframe, expected_df)
    else:
        pytest.skip(f"Expected DataFrame type: {type(expected_df)}")


def test_data_init_kwds() -> None:
    csv_path = csv_dir_path / "test.csv"
    expected_df = pd.read_csv(csv_path, usecols=[0, 1])  # type: ignore
    data = Data(csv_path, usecols=[0, 1])
    assert len(data.dataframe.columns) == 2
    pt.assert_frame_equal(data.dataframe, expected_df)


def test_data_getitem() -> None:
    df = pd.DataFrame([[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=["a", "b", "c"])  # type: ignore
    data = Data(df)

    pt.assert_series_equal(data["a"], df.a)  # type: ignore
    pt.assert_series_equal(data["b"], df.b)  # type: ignore
    pt.assert_series_equal(data["c"], df.c)  # type: ignore


def test_data_getitem_no_header() -> None:
    df = pd.DataFrame([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    data = Data(df)

    pt.assert_series_equal(data[0], df[0])  # type: ignore
    pt.assert_series_equal(data[1], df[1])  # type: ignore
    pt.assert_series_equal(data[2], df[2])  # type: ignore


def test_data_len() -> None:
    df = pd.DataFrame([[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=["a", "b", "c"])  # type: ignore
    data = Data(df)

    assert len(data) == len(df)


def test_data_getattr() -> None:
    df = pd.DataFrame([[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=["a", "b", "c"])  # type: ignore
    data = Data(df)

    pt.assert_index_equal(data.columns, pd.Index(["a", "b", "c"]))
    assert data.shape == (3, 3)
    assert data.to_csv() == ",a,b,c\n0,0,1,2\n1,3,4,5\n2,6,7,8\n"
    assert data.iat[1, 2] == 5
    assert data.at[2, "a"] == 6


def test_data_attributes() -> None:
    df = pd.DataFrame([[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=["a", "b", "c"])  # type: ignore
    data = Data(df)

    pt.assert_series_equal(data.a, df.a)  # type: ignore
    pt.assert_series_equal(data.b, df.b)  # type: ignore
    pt.assert_series_equal(data.c, df.c)  # type: ignore


def test_data_min() -> None:
    csv_path = csv_dir_path / "test.csv"
    data = Data(csv_path)
    assert data.min("a") == 1
    assert data.min("d") == 3.5


def test_data_min_list() -> None:
    csv_path = csv_dir_path / "test.csv"
    data = Data(csv_path)
    assert data.min(["a", "b", "c"]) == [1, 0.01, 10.0]
    assert data.min(["b", "d", "c", "e"]) == [0.01, 3.5, 10.0, 100]


def test_data_max() -> None:
    csv_path = csv_dir_path / "test.csv"
    data = Data(csv_path)
    assert data.max("b") == 0.04
    assert data.max("c") == 40.0


def test_data_max_list() -> None:
    csv_path = csv_dir_path / "test.csv"
    data = Data(csv_path)
    assert data.max(["a", "b", "c"]) == [4, 0.04, 40.0]
    assert data.max(["b", "d", "c", "e"]) == [0.04, 11.5, 40.0, 400]


def test_data_param() -> None:
    csv_path = csv_dir_path / "test.csv"
    data = Data(csv_path)
    assert data.param("b") == 0.01


def test_data_param_list() -> None:
    csv_path = csv_dir_path / "test.csv"
    data = Data(csv_path)
    assert data.param(["c", "e"]) == [10.0, 100]


@pytest.mark.parametrize("cls", [str, Path])
def test_tagged_data_init_path(cls) -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    path = cls(csv_path)
    raw_df = pd.read_csv(csv_path)

    tagged_data = TaggedData(path)

    assert tagged_data.datapath == Path(csv_path)
    assert tagged_data.datadir == Path(csv_dir_path)
    pt.assert_frame_equal(tagged_data.dataframe, raw_df)
    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("tag")
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
    else:
        pytest.skip(f"Expected DataFrame type: {type(raw_df)}")


def test_tagged_data_init_StringIO() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    raw_df = pd.read_csv(csv_path)

    tagged_data = TaggedData(StringIO(test_tagged_data))

    assert tagged_data.datapath is None
    assert tagged_data.datadir is None
    pt.assert_frame_equal(tagged_data.dataframe, raw_df)
    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("tag")
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
    else:
        pytest.skip(f"Expected DataFrame type: {type(raw_df)}")


def test_tagged_data_init_DataFrame() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    raw_df = pd.read_csv(csv_path)

    if isinstance(raw_df, pd.DataFrame):
        tagged_data = TaggedData(raw_df)
        groups = raw_df.groupby("tag")

        assert tagged_data.datapath is None
        assert tagged_data.datadir is None
        pt.assert_frame_equal(tagged_data.dataframe, raw_df)
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
    else:
        pytest.skip(f"Expected DataFrame type: {type(raw_df)}")


def test_tagged_data_non_default_tag() -> None:
    csv_path = csv_dir_path / "test_tagged_data_label.csv"
    raw_df = pd.read_csv(csv_path)

    tagged_data = TaggedData(csv_path, by="label")

    assert tagged_data.datapath == Path(csv_path)
    assert tagged_data.datadir == Path(csv_dir_path)
    pt.assert_frame_equal(tagged_data.dataframe, raw_df)
    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("label")
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
    else:
        pytest.skip(f"Expected DataFrame type: {type(raw_df)}")


def test_tagged_data_no_tag() -> None:
    csv_path = csv_dir_path / "test.csv"
    raw_df = pd.read_csv(csv_path)

    tagged_data = TaggedData(csv_path)
    pt.assert_frame_equal(tagged_data.dataframe, raw_df)
    if isinstance(raw_df, pd.DataFrame):
        assert len(tagged_data.datadict) == 1
        pt.assert_frame_equal(tagged_data.datadict["0"].dataframe, raw_df)
    else:
        pytest.skip(f"Expected DataFrame type: {type(raw_df)}")


def test_tagged_data_iter() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    tagged_data = TaggedData(csv_path)
    raw_df = pd.read_csv(csv_path)

    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("tag")
        for i, data in enumerate(tagged_data):
            pt.assert_frame_equal(
                data.dataframe,
                groups.get_group(f"tag{i+1:02d}").reset_index(drop=True),
            )
    else:
        pytest.skip(f"Expected DataFrame type: {type(raw_df)}")


def test_tagged_data_property_datadict() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    tagged_data = TaggedData(csv_path)
    assert isinstance(tagged_data.datadict, dict)
    assert list(tagged_data.datadict.keys()) == ["tag01", "tag02", "tag03"]


def test_tagged_data_keys() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    tagged_data = TaggedData(csv_path)
    assert tagged_data.keys() == ["tag01", "tag02", "tag03"]


def test_tagged_data_tags() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    tagged_data = TaggedData(csv_path)
    assert tagged_data.tags() == ["tag01", "tag02", "tag03"]


def test_tagged_data_items() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    tagged_data = TaggedData(csv_path)
    raw_df = pd.read_csv(csv_path)

    items = tagged_data.items()

    assert isinstance(items, list)
    assert len(items) == 3
    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("tag")
        for i, tup in enumerate(items):
            tag = tup[0]
            data = tup[1]
            assert tag == f"tag{i+1:02d}"
            pt.assert_frame_equal(
                data.dataframe,
                groups.get_group(tag).reset_index(drop=True),
            )
    else:
        pytest.skip(f"Expected DataFrame type: {type(raw_df)}")


def test_tagged_data_get() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    tagged_data = TaggedData(csv_path)
    raw_df = pd.read_csv(csv_path)

    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("tag")
        for tag in ["tag01", "tag02", "tag03"]:
            data = tagged_data.get(tag)
            pt.assert_frame_equal(
                data.dataframe,
                groups.get_group(tag).reset_index(drop=True),
            )
    else:
        pytest.skip(f"Expected DataFrame type: {type(raw_df)}")


def test_tagged_data_get_default() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    tagged_data = TaggedData(csv_path)
    raw_df = pd.read_csv(csv_path)

    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("tag")
        data = tagged_data.get()
        pt.assert_frame_equal(
            data.dataframe,
            groups.get_group("tag01").reset_index(drop=True),
        )
    else:
        pytest.skip(f"Expected DataFrame type: {type(raw_df)}")


def test_tagged_data_param() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    tagged_data = TaggedData(csv_path)

    assert tagged_data.param("a", "tag01") == 0
    assert tagged_data.param("b", "tag02") == 11
    assert tagged_data.param("c", "tag03") == 22


def test_tagged_data_param_default() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    tagged_data = TaggedData(csv_path)

    assert tagged_data.param("a") == 0
    assert tagged_data.param("b") == 1
    assert tagged_data.param("c") == 2


def test_tagged_data_param_list() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    tagged_data = TaggedData(csv_path)

    assert tagged_data.param(["a", "b", "c"], "tag01") == [0, 1, 2]
    assert tagged_data.param(["b", "c", "d"], "tag02") == [11, 12, 13]
    assert tagged_data.param(["c", "d", "e"], "tag03") == [22, 23, 24]


def test_tagged_data_param_list_split() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    tagged_data = TaggedData(csv_path)

    a, b, c = tagged_data.param(["a", "b", "c"], "tag01")

    assert a == 0
    assert b == 1
    assert c == 2


def test_tagged_data_param_list_default() -> None:
    csv_path = csv_dir_path / "test_tagged_data.csv"
    tagged_data = TaggedData(csv_path)

    assert tagged_data.param(["a", "b", "c", "d", "e"]) == [0, 1, 2, 3, 4]


# Local Variables:
# jinx-local-words: "cls csv noqa"
# End:
