"""Data Handling and Manipulation Module.

This module provides classes for managing and manipulating tabular data, with functionalities to
load data from various sources, grouped data structure by specified tags, and access columns or rows
with intuitive syntax. The primary classes, `Data`, and `TaggedData`, facilitate working with
tabular data in pandas DataFrame while allowing access to specific features like data grouping,
dynamic attribute setting, and easy retrieval of minimum, maximum, and parameter values.

Classes
-------
BaseData : Abstract base class providing the core attributes and methods for data handling.
    Defines basic properties for data path and DataFrame storage.

Data : Extends BaseData to represent a single tabular data.
    Provides methods to access columns and calculate minimum, maximum, and specific parameters.

TaggedData : Extends BaseData to handle grouped data based on a specified tag column.
    Allows grouping data by a tag and accessing each group as a separate `Data` object.

Examples
--------
Basic usage:
    >>> data = Data("data.csv")
    >>> print(data.min("column_name"))

Tagged data usage:
    >>> tagged_data = TaggedData("data.csv", by="tag")
    >>> group = tagged_data.get("specific_tag")
    >>> print(group.max("column_name"))

This module is designed to streamline operations with tabular data in data analysis, data plotting,
and other applications requiring structured data handling.

"""

from __future__ import annotations

import os
from collections.abc import Hashable, Iterator, Sequence
from io import FileIO, StringIO, TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

if TYPE_CHECKING:
    from pandas.io.parsers import TextFileReader
    from pandas.io.parsers.readers import UsecolsArgType

FilePath: TypeAlias = str | Path
DataSourceType: TypeAlias = FilePath | StringIO | pd.DataFrame
NumericType: TypeAlias = int | float | complex | np.number
NumericTypeVar = TypeVar("NumericTypeVar", bound=NumericType)


class BaseData:
    """Base class for data handling and manipulation.

    This class has functionalities for setting and retrieving the path and the main DataFrame
    associated with the data.

    """

    _datapath: Path
    _dataframe: pd.DataFrame

    def __init__(
        self,
        data_source: DataSourceType,
        *,
        sep: str,
        header: int | Sequence[int] | Literal["infer"] | None,
        names: Sequence[Hashable] | None,
        usecols: UsecolsArgType,
        nrows: int | None,
        comment: str | None,
    ) -> None:
        """Initialize the BaseData object with the provided data source.

        Parameters
        ----------
        data : str, Path, StringIO, or pd.DataFrame
            The data source.
        **kwds : dict, optional
            Additional keyword arguments passed to `pd.read_csv`.

        Raises
        ------
        TypeError
            If the data type is unsupported.

        """
        if isinstance(data_source, pd.DataFrame):
            self._set_dataframe(data_source)
        elif isinstance(data_source, StringIO | FilePath):
            self._set_dataframe(
                self.read_csv(
                    data_source,
                    sep=sep,
                    header=header,
                    names=names,
                    usecols=usecols,
                    nrows=nrows,
                    comment=comment,
                ),
            )
            if isinstance(data_source, FilePath):
                self._set_datapath(data_source)
        else:
            msg = f"Unsupported data source type: {type(data_source)}"
            raise TypeError(msg)

    @staticmethod
    def read_commented_column_names(file_or_buffer: FilePath | StringIO, *, sep: str, comment: str) -> list[str] | None:
        """Return a list of column names.

        File or string buffer are assumed to start lines with commented lines. The last commented
        line is split with `sep`. The split strings are returned as a list of column names.

        """

        def last_commented_header(buffer: TextIOWrapper, comment: str) -> str:
            header = ""
            for line in buffer:
                if line.startswith(comment):
                    header = line
                else:
                    break
            return header

        if isinstance(file_or_buffer, FilePath):
            with Path(file_or_buffer).open() as f:
                header = last_commented_header(f, comment)
        else:
            header = last_commented_header(file_or_buffer, comment)
            file_or_buffer.seek(0)
        if len(header) > 0:
            return header[1:].strip().split(sep)
        return None

    @staticmethod
    def read_csv(
        file_or_buffer: FilePath | StringIO,
        *,
        sep: str = ",",
        header: int | Sequence[int] | Literal["infer"] | None,
        names: Sequence[Hashable] | None,
        usecols: UsecolsArgType,
        nrows: int | None,
        comment: str | None,
    ) -> pd.DataFrame:
        """Return a pandas DataFrame loaded from a file or string buffer."""
        if comment is not None and names is None:
            names = BaseData.read_commented_column_names(file_or_buffer, sep=sep, comment=comment)
        return pd.read_csv(
            file_or_buffer,
            sep=sep,
            header=header,
            names=names,
            usecols=usecols,
            nrows=nrows,
            comment=comment,
            iterator=False,
            chunksize=None,
        )

    def _set_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Set the DataFrame associated with the data object.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame to associate with the data.

        Raises
        ------
        TypeError
            If the provided df is not a DataFrame.

        """
        self._dataframe = dataframe

    @property
    def dataframe(self) -> pd.DataFrame:
        """Retrieve the raw DataFrame associated with the data.

        Returns
        -------
        pd.DataFrame
            The DataFrame associated with the data.

        """
        return self._dataframe

    def is_loaded_from_file(self) -> bool:
        try:
            _ = self._datapath
        except AttributeError:
            return False
        return True

    def _set_datapath(self, datapath: str | Path) -> None:
        """Set the path to the data file.

        Parameters
        ----------
        datapath : str or Path
            Path to the data file.

        """
        self._datapath = Path(datapath)

    @property
    def datapath(self) -> Path:
        """Retrieve the path to the data file.

        Returns
        -------
        Path or None
            Path to the data file, if set; otherwise, None.

        """
        return self._datapath

    @property
    def datadir(self) -> Path | None:
        """Retrieve the directory of the data file.

        Returns
        -------
        Path or None
            Directory of the data file, if the path is set; otherwise, None.

        """
        if isinstance(self._datapath, Path):
            return self._datapath.parent
        return None

    def __str__(self) -> str:
        """Return a string representation of the DataFrame.

        Returns
        -------
        str
            String representation of the DataFrame.

        """
        return str(self._dataframe)


class Data(BaseData):
    """Class for handling and manipulating tabular data.

    This class has methods for accessing, retrieving, and setting data attributes.

    """

    def __init__(
        self,
        data_source: DataSourceType,
        *,
        sep: str = ",",
        header: int | Sequence[int] | Literal["infer"] | None = "infer",
        names: Sequence[Hashable] | None = None,
        usecols: UsecolsArgType = None,
        nrows: int | None = None,
        comment: str | None = None,
    ) -> None:
        """Initialize the Data object with the provided data source.

        Parameters
        ----------
        data : str, Path, StringIO, or pd.DataFrame
            The data source.
        **kwds : dict, optional
            Additional keyword arguments passed to `pd.read_csv`.

        Raises
        ------
        TypeError
            If the data type is unsupported.

        """
        super().__init__(
            data_source,
            sep=sep,
            header=header,
            names=names,
            usecols=usecols,
            nrows=nrows,
            comment=comment,
        )

        # self._set_attributes()

    # def __getitem__(self, key: str | int) -> pd.Series:
    #     """Access a specific column or row by key.

    #     Parameters
    #     ----------
    #     key : str or int
    #         Column name or row index.

    #     Returns
    #     -------
    #     pd.Series
    #         The column or row data as a Series.

    #     """
    #     return self.dataframe[key]

    def __len__(self) -> int:
        """Return the number of rows in the DataFrame.

        Returns
        -------
        int
            Number of rows in the DataFrame.

        """
        return len(self.dataframe)

    def __getattr__(self, name: str):  # noqa: ANN204
        """Access DataFrame attributes not explicitly defined in Data.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        Any
            The attribute from the DataFrame.

        """
        return getattr(self.dataframe, name)

    def _set_attributes(self) -> None:
        """Set column names as attributes for quick access if column names are strings."""
        if is_string_dtype(self.dataframe.columns):
            for c in self.dataframe.columns:
                setattr(self, str(c), getattr(self.dataframe, str(c)))

    @overload
    def min(self, col: str) -> NumericType: ...

    @overload
    def min(self, col: list[str]) -> list[NumericType]: ...

    def min(self, col):
        """Compute the minimum value of the specified column(s).

        Parameters
        ----------
        col : str or list of str
            Column name or list of column names.

        Returns
        -------
        NumericType or list of NumericType
            Minimum value(s).

        """
        if isinstance(col, str):
            return self.dataframe[col].min()
        if isinstance(col, Sequence):
            return [self.dataframe[c].min() for c in col]

        msg = f"unsupported type: {type(col)}"
        raise TypeError(msg)

    @overload
    def max(self, col: str) -> NumericType: ...

    @overload
    def max(self, col: list[str]) -> list[NumericType]: ...

    def max(self, col):
        """Compute the maximum value of the specified column(s).

        Parameters
        ----------
        col : str or list of str
            Column name or list of column names.

        Returns
        -------
        NumericType or list of NumericType
            Maximum value(s).

        """
        if isinstance(col, str):
            return self.dataframe[col].max()
        if isinstance(col, Sequence):
            return [self.dataframe[c].max() for c in col]

        msg = f"unsupported type: {type(col)}"
        raise TypeError(msg)

    @overload
    def param(self, col: str) -> NumericType: ...

    @overload
    def param(self, col: list[str] | tuple[str]) -> list[NumericType]: ...

    def param(self, col):
        """Retrieve the first value(s) of the specified column(s).

        Parameters
        ----------
        col : str, list of str, or tuple of str
            Column name or list of column names.

        Returns
        -------
        NumericType or list of NumericType
            First value(s) in the column(s).

        """
        if isinstance(col, str):
            return self.dataframe.loc[0, col]
        if isinstance(col, Sequence):
            return [self.dataframe.loc[0, c] for c in col]

        msg = f"unsupported type: {type(col)}"
        raise TypeError(msg)


class TaggedData(BaseData):
    """Class for managing data tagged by a specific column for easy access by group."""

    _datadict: dict[str, Data]
    _groups: Any
    _by: str

    # def __init__(self, data_source: str | Path | StringIO | pd.DataFrame, **kwds) -> None:  # noqa: ANN003
    #     """Initialize the TaggedData object and groups data by the specified tag.

    #     Parameters
    #     ----------
    #     data : str, Path, StringIO, or pd.DataFrame
    #         The data source.
    #     **kwds : dict, optional
    #         Additional keyword arguments passed to `pd.read_csv`.

    #     """
    #     super().__init__(data_source)
    #     self._datadict = {}
    #     self._by = kwds.pop("by", "tag")

    #     if isinstance(data, str | Path):
    #         self._set_datapath(data)

    #     if self.datapath is not None:
    #         self._set_dataframe(pd.read_csv(self.datapath, **kwds))
    #     elif isinstance(data, StringIO):
    #         self._set_dataframe(pd.read_csv(data, **kwds))
    #     elif isinstance(data, pd.DataFrame):
    #         self._set_dataframe(data)
    #     else:
    #         msg = f"unsupported type: {type(data)}"
    #         raise TypeError(msg)

    #     self._make_groups()

    # def __iter__(self) -> Iterator[Data]:
    #     """Return an iterator over the grouped Data objects.

    #     Returns
    #     -------
    #     Iterator[Data]
    #         An iterator over the grouped Data objects.

    #     """
    #     return iter(self._datadict.values())

    # def _make_groups(self) -> None:
    #     """Group the data by the specified tag and stores it in `datadict`."""
    #     if self._by in self.dataframe.columns:
    #         self._groups = self.dataframe.groupby(self._by)
    #         self._datadict = {
    #             str(k): Data(self._groups.get_group(k).reset_index(drop=True)) for k in self._groups.groups
    #         }
    #     else:
    #         self._datadict = {"0": Data(self.dataframe)}

    # @property
    # def datadict(self) -> dict[str, Data]:
    #     """Retrieve the dictionary of grouped Data objects.

    #     Returns
    #     -------
    #     dict of str, Data
    #         Dictionary of grouped Data objects.

    #     """
    #     return self._datadict

    # def keys(self) -> list[str]:
    #     """Return the tags associated with the data groups.

    #     Returns
    #     -------
    #     list of str
    #         List of tags.

    #     """
    #     return list(self.datadict.keys())

    # def tags(self) -> list[str]:
    #     """Return the tags associated with the data groups.

    #     Returns
    #     -------
    #     list of str
    #         List of tags.

    #     """
    #     return self.keys()

    # def items(self) -> list[tuple[str, Data]]:
    #     """Retrieve the items (tag and Data object) of the grouped data.

    #     Returns
    #     -------
    #     list of tuple of (str, Data)
    #         List of tag-Data object pairs.

    #     """
    #     return list(self.datadict.items())

    # def get(self, tag: str | None = None) -> Data:
    #     """Retrieve the Data object associated with the specified tag.

    #     Parameters
    #     ----------
    #     tag : str or None, optional
    #         Tag of the data group to retrieve.

    #     Returns
    #     -------
    #     Data
    #         Data object corresponding to the tag.

    #     """
    #     if tag is None:
    #         tag = self.keys()[0]
    #     return self.datadict[tag]

    # @overload
    # def param(self, col: str, tag: str | None = None) -> NumericType: ...

    # @overload
    # def param(self, col: list[str] | tuple[str], tag: str | None = None) -> list[NumericType]: ...

    # def param(self, col, tag=None):
    #     """Retrieve the first value(s) of the specified column(s) from a tagged Data object.

    #     Parameters
    #     ----------
    #     col : str, list of str, or tuple of str
    #         Column name or list of column names.
    #     tag : str or None, optional
    #         Tag of the data group to retrieve.

    #     Returns
    #     -------
    #     NumericType or list of NumericType
    #         First value(s) in the column(s).

    #     """
    #     if tag is None:
    #         tag = self.keys()[0]

    #     if isinstance(col, str):
    #         return self.datadict[tag].dataframe.loc[0, col]
    #     if isinstance(col, Sequence):
    #         return [self.datadict[tag].dataframe.loc[0, c] for c in col]

    #     msg = f"unsupported type: {type(col)}"
    #     raise TypeError(msg)


# Local Variables:
# jinx-local-words: "StringIO csv datadict datapath df kwds noqa str"
# End:
