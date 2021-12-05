import os
from io import StringIO
from pathlib import Path

import pandas as pd
import pandas._testing as pt
import pytest
from pyplotutil.datautil import Data

csv_dir_path = os.path.join(os.path.dirname(__file__), "data")

test_data = """\
a,b,c,d,e
1,0.01,10.0,3.5,100
2,0.02,20.0,7.5,200
3,0.03,30.0,9.5,300
4,0.04,40.0,11.5,400
"""


@pytest.mark.parametrize("cls", [str, Path])
def test_data_init_path(cls) -> None:
    csv_path = os.path.join(csv_dir_path, "test.csv")
    path = cls(csv_path)
    expected_df = pd.read_csv(csv_path)

    data = Data(path)

    assert data.datapath == Path(csv_path)
    pt.assert_frame_equal(data.dataframe, expected_df)


def test_data_init_StringIO() -> None:
    csv_path = os.path.join(csv_dir_path, "test.csv")
    expected_df = pd.read_csv(csv_path)

    data = Data(StringIO(test_data))

    assert data.datapath is None
    pt.assert_frame_equal(data.dataframe, expected_df)


def test_data_init_DataFrame() -> None:
    csv_path = os.path.join(csv_dir_path, "test.csv")
    expected_df = pd.read_csv(csv_path)

    if isinstance(expected_df, pd.DataFrame):
        data = Data(expected_df)

        assert data.datapath is None
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
