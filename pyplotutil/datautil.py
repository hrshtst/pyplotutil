# -*- coding: utf-8 -*-

from io import StringIO
from pathlib import Path
from typing import Any, Iterator, Sequence, Tuple, overload

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.io.parsers import TextFileReader

NumericType = int | float | complex | np.number


class BaseData(object):
    _datapath: Path | None
    _dataframe: pd.DataFrame

    def __init__(self) -> None:
        self._datapath = None

    def _set_datapath(self, datapath: str | Path) -> None:
        self._datapath = Path(datapath)

    def _get_datapath(self) -> Path | None:
        return self._datapath

    datapath = property(_get_datapath, _set_datapath)

    def _get_datadir(self) -> Path | None:
        if isinstance(self._datapath, Path):
            return self._datapath.parent
        return None

    datadir = property(_get_datadir)

    def _set_dataframe(self, df: pd.DataFrame | TextFileReader) -> None:
        if isinstance(df, pd.DataFrame):
            self._dataframe = df
        else:
            raise TypeError(f"unsupported type: {type(df)}")

    def _get_dataframe(self) -> pd.DataFrame:
        return self._dataframe

    dataframe = property(_get_dataframe, _set_dataframe)

    def __str__(self) -> str:
        return str(self._dataframe)


class Data(BaseData):
    def __init__(self, data: str | Path | StringIO | pd.DataFrame, **kwds) -> None:
        super().__init__()

        if isinstance(data, (str, Path)):
            self._set_datapath(data)

        if self.datapath is not None:
            self._set_dataframe(pd.read_csv(self.datapath, **kwds))
        elif isinstance(data, StringIO):
            self._set_dataframe(pd.read_csv(data, **kwds))
        elif isinstance(data, pd.DataFrame):
            self._set_dataframe(data)
        else:
            raise TypeError(f"unsupported type: {type(data)}")

        self._set_attributes()

    def __getitem__(self, key: str | int) -> pd.Series:
        return self.dataframe[key]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getattr__(self, name: str):
        return getattr(self.dataframe, name)

    def _set_attributes(self) -> None:
        if is_string_dtype(self.dataframe.columns):
            for c in self.dataframe.columns:
                setattr(self, str(c), getattr(self.dataframe, str(c)))

    @overload
    def param(self, col: str) -> NumericType:
        ...

    @overload
    def param(self, col: list[str] | tuple[str]) -> list[NumericType]:
        ...

    def param(self, col):
        if isinstance(col, str):
            return self.dataframe.at[0, col]
        elif isinstance(col, Sequence):
            return [self.dataframe.at[0, c] for c in col]
        else:
            raise TypeError(f"unsupported type: {type(col)}")


class DataSet(BaseData):
    _datadict: dict[str, Data]
    _groups: Any
    _by: str

    def __init__(self, data: str | Path | StringIO | pd.DataFrame, **kwds) -> None:
        super().__init__()
        self._datadict = {}
        self._by = kwds.pop("by", "tag")

        if isinstance(data, (str, Path)):
            self._set_datapath(data)

        if self.datapath is not None:
            self._set_dataframe(pd.read_csv(self.datapath, **kwds))
        elif isinstance(data, StringIO):
            self._set_dataframe(pd.read_csv(data, **kwds))
        elif isinstance(data, pd.DataFrame):
            self._set_dataframe(data)
        else:
            raise TypeError(f"unsupported type: {type(data)}")

        self._make_groups()

    def __iter__(self) -> Iterator[Data]:
        return iter(self._datadict.values())

    def _make_groups(self):
        if self._by in self.dataframe.columns:
            self._groups = self.dataframe.groupby(self._by)
            self._datadict = {
                str(k): Data(self._groups.get_group(k).reset_index(drop=True))
                for k in self._groups.groups.keys()
            }
        else:
            self._datadict = {"0": Data(self.dataframe)}

    @property
    def datadict(self) -> dict[str, Data]:
        return self._datadict

    def keys(self) -> list[str]:
        return list(self.datadict.keys())

    def items(self) -> list[Tuple[str, Data]]:
        return list(self.datadict.items())

    def get(self, tag: str | None = None) -> Data:
        if tag is None:
            tag = self.keys()[0]
        return self.datadict[tag]

    @overload
    def param(self, col: str, tag: str | None = None) -> NumericType:
        ...

    @overload
    def param(
        self, col: list[str] | tuple[str], tag: str | None = None
    ) -> list[NumericType]:
        ...

    def param(self, col, tag=None):
        if tag is None:
            tag = self.keys()[0]

        if isinstance(col, str):
            return self.datadict[tag].dataframe.at[0, col]
        elif isinstance(col, Sequence):
            return [self.datadict[tag].dataframe.at[0, c] for c in col]
        else:
            raise TypeError(f"unsupported type: {type(col)}")
