import os
from io import StringIO
from pathlib import Path

import pandas as pd
import pandas._testing as pt
import pytest
from pyplotutil.datautil import Data, DataSet

csv_dir_path = os.path.join(os.path.dirname(__file__), "data")

test_data = """\
a,b,c,d,e
1,0.01,10.0,3.5,100
2,0.02,20.0,7.5,200
3,0.03,30.0,9.5,300
4,0.04,40.0,11.5,400
"""

test_dataset = """\
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
    csv_path = os.path.join(csv_dir_path, "test.csv")
    path = cls(csv_path)
    expected_df = pd.read_csv(csv_path)

    data = Data(path)

    assert data.datapath == Path(csv_path)
    assert data.datadir == Path(csv_dir_path)
    pt.assert_frame_equal(data.dataframe, expected_df)


def test_data_init_StringIO() -> None:
    csv_path = os.path.join(csv_dir_path, "test.csv")
    expected_df = pd.read_csv(csv_path)

    data = Data(StringIO(test_data))

    assert data.datapath is None
    assert data.datadir is None
    pt.assert_frame_equal(data.dataframe, expected_df)


def test_data_init_DataFrame() -> None:
    csv_path = os.path.join(csv_dir_path, "test.csv")
    expected_df = pd.read_csv(csv_path)

    if isinstance(expected_df, pd.DataFrame):
        data = Data(expected_df)

        assert data.datapath is None
        assert data.datadir is None
        pt.assert_frame_equal(data.dataframe, expected_df)
    else:
        pytest.skip(f"Expected DataFram type: {type(expected_df)}")


def test_data_init_kwds() -> None:
    csv_path = os.path.join(csv_dir_path, "test.csv")
    expected_df = pd.read_csv(csv_path, usecols=[0, 1])
    data = Data(csv_path, usecols=[0, 1])
    assert len(data.dataframe.columns) == 2
    pt.assert_frame_equal(data.dataframe, expected_df)


def test_data_getitem() -> None:
    df = pd.DataFrame([[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=["a", "b", "c"])
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
    df = pd.DataFrame([[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=["a", "b", "c"])
    data = Data(df)

    assert len(data) == len(df)


def test_data_getattr() -> None:
    df = pd.DataFrame([[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=["a", "b", "c"])
    data = Data(df)

    pt.assert_index_equal(data.columns, pd.Index(["a", "b", "c"]))
    assert data.shape == (3, 3)
    assert data.to_csv() == ",a,b,c\n0,0,1,2\n1,3,4,5\n2,6,7,8\n"
    assert data.iat[1, 2] == 5
    assert data.at[2, "a"] == 6


def test_data_attributes() -> None:
    df = pd.DataFrame([[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=["a", "b", "c"])
    data = Data(df)

    pt.assert_series_equal(data.a, df.a)  # type: ignore
    pt.assert_series_equal(data.b, df.b)  # type: ignore
    pt.assert_series_equal(data.c, df.c)  # type: ignore


def test_data_param() -> None:
    csv_path = os.path.join(csv_dir_path, "test.csv")
    data = Data(csv_path)
    assert data.param("b") == 0.01


def test_data_param_list() -> None:
    csv_path = os.path.join(csv_dir_path, "test.csv")
    data = Data(csv_path)
    assert data.param(["c", "e"]) == [10.0, 100]


@pytest.mark.parametrize("cls", [str, Path])
def test_dataset_init_path(cls) -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    path = cls(csv_path)
    raw_df = pd.read_csv(csv_path)

    dataset = DataSet(path)

    assert dataset.datapath == Path(csv_path)
    assert dataset.datadir == Path(csv_dir_path)
    pt.assert_frame_equal(dataset.dataframe, raw_df)
    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("tag")
        datadict = dataset._datadict
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
        pytest.skip(f"Expected DataFram type: {type(raw_df)}")


def test_dataset_init_StringIO() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    raw_df = pd.read_csv(csv_path)

    dataset = DataSet(StringIO(test_dataset))

    assert dataset.datapath is None
    assert dataset.datadir is None
    pt.assert_frame_equal(dataset.dataframe, raw_df)
    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("tag")
        datadict = dataset._datadict
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
        pytest.skip(f"Expected DataFram type: {type(raw_df)}")


def test_dataset_init_DataFrame() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    raw_df = pd.read_csv(csv_path)

    if isinstance(raw_df, pd.DataFrame):
        dataset = DataSet(raw_df)
        groups = raw_df.groupby("tag")

        assert dataset.datapath is None
        assert dataset.datadir is None
        pt.assert_frame_equal(dataset.dataframe, raw_df)
        pt.assert_frame_equal(
            dataset._datadict["tag01"].dataframe,
            groups.get_group("tag01").reset_index(drop=True),
        )
        pt.assert_frame_equal(
            dataset._datadict["tag02"].dataframe,
            groups.get_group("tag02").reset_index(drop=True),
        )
        pt.assert_frame_equal(
            dataset._datadict["tag03"].dataframe,
            groups.get_group("tag03").reset_index(drop=True),
        )
    else:
        pytest.skip(f"Expected DataFram type: {type(raw_df)}")


def test_dataset_non_default_tag() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset_label.csv")
    raw_df = pd.read_csv(csv_path)

    dataset = DataSet(csv_path, by="label")

    assert dataset.datapath == Path(csv_path)
    assert dataset.datadir == Path(csv_dir_path)
    pt.assert_frame_equal(dataset.dataframe, raw_df)
    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("label")
        pt.assert_frame_equal(
            dataset._datadict["label01"].dataframe,
            groups.get_group("label01").reset_index(drop=True),
        )
        pt.assert_frame_equal(
            dataset._datadict["label02"].dataframe,
            groups.get_group("label02").reset_index(drop=True),
        )
        pt.assert_frame_equal(
            dataset._datadict["label03"].dataframe,
            groups.get_group("label03").reset_index(drop=True),
        )
    else:
        pytest.skip(f"Expected DataFram type: {type(raw_df)}")


def test_dataset_no_tag() -> None:
    csv_path = os.path.join(csv_dir_path, "test.csv")
    raw_df = pd.read_csv(csv_path)

    dataset = DataSet(csv_path)
    pt.assert_frame_equal(dataset.dataframe, raw_df)
    if isinstance(raw_df, pd.DataFrame):
        assert len(dataset.datadict) == 1
        pt.assert_frame_equal(dataset.datadict["0"].dataframe, raw_df)
    else:
        pytest.skip(f"Expected DataFram type: {type(raw_df)}")


def test_dataset_iter() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    dataset = DataSet(csv_path)
    raw_df = pd.read_csv(csv_path)

    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("tag")
        for i, data in enumerate(dataset):
            pt.assert_frame_equal(
                data.dataframe, groups.get_group(f"tag{i+1:02d}").reset_index(drop=True)
            )
    else:
        pytest.skip(f"Expected DataFram type: {type(raw_df)}")


def test_dataset_property_datadict() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    dataset = DataSet(csv_path)
    assert isinstance(dataset.datadict, dict)
    assert list(dataset.datadict.keys()) == ["tag01", "tag02", "tag03"]


def test_dataset_keys() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    dataset = DataSet(csv_path)
    assert dataset.keys() == ["tag01", "tag02", "tag03"]


def test_dataset_tags() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    dataset = DataSet(csv_path)
    assert dataset.tags() == ["tag01", "tag02", "tag03"]


def test_dataset_items() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    dataset = DataSet(csv_path)
    raw_df = pd.read_csv(csv_path)

    items = dataset.items()

    assert isinstance(items, list)
    assert len(items) == 3
    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("tag")
        for i, tup in enumerate(items):
            tag = tup[0]
            data = tup[1]
            assert tag == f"tag{i+1:02d}"
            pt.assert_frame_equal(
                data.dataframe, groups.get_group(tag).reset_index(drop=True)
            )
    else:
        pytest.skip(f"Expected DataFram type: {type(raw_df)}")


def test_dataset_get() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    dataset = DataSet(csv_path)
    raw_df = pd.read_csv(csv_path)

    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("tag")
        for tag in ["tag01", "tag02", "tag03"]:
            data = dataset.get(tag)
            pt.assert_frame_equal(
                data.dataframe, groups.get_group(tag).reset_index(drop=True)
            )
    else:
        pytest.skip(f"Expected DataFram type: {type(raw_df)}")


def test_dataset_get_default() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    dataset = DataSet(csv_path)
    raw_df = pd.read_csv(csv_path)

    if isinstance(raw_df, pd.DataFrame):
        groups = raw_df.groupby("tag")
        data = dataset.get()
        pt.assert_frame_equal(
            data.dataframe, groups.get_group("tag01").reset_index(drop=True)
        )
    else:
        pytest.skip(f"Expected DataFram type: {type(raw_df)}")


def test_dataset_param() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    dataset = DataSet(csv_path)

    assert dataset.param("a", "tag01") == 0
    assert dataset.param("b", "tag02") == 11
    assert dataset.param("c", "tag03") == 22


def test_dataset_param_default() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    dataset = DataSet(csv_path)

    assert dataset.param("a") == 0
    assert dataset.param("b") == 1
    assert dataset.param("c") == 2


def test_dataset_param_list() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    dataset = DataSet(csv_path)

    assert dataset.param(["a", "b", "c"], "tag01") == [0, 1, 2]
    assert dataset.param(["b", "c", "d"], "tag02") == [11, 12, 13]
    assert dataset.param(["c", "d", "e"], "tag03") == [22, 23, 24]


def test_dataset_param_list_split() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    dataset = DataSet(csv_path)

    a, b, c = dataset.param(["a", "b", "c"], "tag01")

    assert a == 0
    assert b == 1
    assert c == 2


def test_dataset_param_list_default() -> None:
    csv_path = os.path.join(csv_dir_path, "test_dataset.csv")
    dataset = DataSet(csv_path)

    assert dataset.param(["a", "b", "c", "d", "e"]) == [0, 1, 2, 3, 4]
