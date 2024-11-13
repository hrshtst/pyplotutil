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

from enum import Enum, auto
from io import StringIO, TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, TypeVar, overload

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Hashable, ItemsView, Iterator, KeysView, Sequence

    from pandas.io.parsers.readers import UsecolsArgType

FilePath: TypeAlias = str | Path
DataSourceType: TypeAlias = FilePath | StringIO | pd.DataFrame | pd.Series
NumericType: TypeAlias = int | float | complex | np.number
NumericTypeVar = TypeVar("NumericTypeVar", bound=NumericType)
Unknown: TypeAlias = Any


class _NoDefault(Enum):
    no_default = auto()


no_default: Final = _NoDefault.no_default
NoDefault: TypeAlias = Literal[_NoDefault.no_default]


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
        elif isinstance(data_source, pd.Series):
            self._set_dataframe(data_source.to_frame())
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
        if isinstance(file_or_buffer, StringIO):
            file_or_buffer.seek(0)
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

    @property
    def df(self) -> pd.DataFrame:
        """Alias for `dataframe` attribute.

        Returns
        -------
        pd.DataFrame
            The DataFrame associated with the data.

        """
        return self.dataframe

    def is_loaded_from_file(self) -> bool:
        """Check if `Data` object is loaded from a file."""
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
        try:
            return self._datapath
        except AttributeError as e:
            msg = "Data object may not be loaded from a file."
            raise AttributeError(msg) from e

    @property
    def datadir(self) -> Path:
        """Retrieve the directory of the data file.

        Returns
        -------
        Path or None
            Directory of the data file, if the path is set; otherwise, None.

        """
        return self.datapath.parent

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

    def __getitem__(self, key: Unknown) -> pd.Series | pd.DataFrame:
        """Access a specific column or row by key.

        Parameters
        ----------
        key : str or int
            Column name or row index.

        Returns
        -------
        pd.Series
            The column or row data as a Series.

        """
        return self.dataframe.__getitem__(key)

    def __len__(self) -> int:
        """Return the number of rows in the DataFrame.

        Returns
        -------
        int
            Number of rows in the DataFrame.

        """
        return len(self.dataframe)

    def __getattr__(self, name: str) -> Unknown:
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
        if name in ("datapath", "datadir"):
            return self.__getattribute__(name)
        return getattr(self.dataframe, name)

    @overload
    def param(self, key: int | str) -> NumericType: ...

    @overload
    def param(self, key: Sequence) -> pd.Series: ...

    def param(self, key):
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
        row = self.dataframe.loc[0, key]
        if isinstance(row, pd.Series | pd.DataFrame):
            row = pd.to_numeric(row)
        return row


class TaggedData(BaseData):
    """Class for managing data tagged by a specific column for easy access by group."""

    _datadict: dict[str, Data]

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
        tag: Unknown = "tag",
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
        self._make_groups(tag)

    def __iter__(self) -> Iterator[Data]:
        """Return an iterator over the grouped Data objects.

        Returns
        -------
        Iterator[Data]
            An iterator over the grouped Data objects.

        """
        return iter(self.datadict.values())

    def _make_groups(self, by: Unknown) -> None:
        """Group the data by the specified tag and stores it in `datadict`."""
        self._datadict = {}
        try:
            groups = self.dataframe.groupby(by)
        except KeyError:
            self._datadict = {"unknown": Data(self.dataframe)}
        else:
            self._datadict = {str(k): Data(groups.get_group(k).reset_index(drop=True)) for k in groups.groups}

    @property
    def datadict(self) -> dict[str, Data]:
        """Retrieve the dictionary of grouped Data objects.

        Returns
        -------
        dict of str, Data
            Dictionary of grouped Data objects.

        """
        return self._datadict

    def tags(self) -> KeysView[str]:
        """Return the tags associated with the data groups.

        Returns
        -------
        list of str
            List of tags.

        """
        return self.datadict.keys()

    def items(self) -> ItemsView[str, Data]:
        """Retrieve the items (tag and Data object) of the grouped data.

        Returns
        -------
        list of tuple of (str, Data)
            List of tag-Data object pairs.

        """
        return self.datadict.items()

    @overload
    def get(self, tag: str) -> Data: ...

    @overload
    def get(self, tag: str, default: Data | NoDefault) -> Data: ...

    @overload
    def get(self, tag: str, default: None) -> Data | None: ...

    def get(self, tag, default=no_default):
        """Retrieve the Data object associated with the specified tag.

        Parameters
        ----------
        tag : str or None, optional
            Tag of the data group to retrieve.

        Returns
        -------
        Data
            Data object corresponding to the tag.

        """
        if default is no_default:
            return self.datadict[tag]
        if default is None:
            # note: this return statement seems nonsense, but without it type checker produces an
            # error, somehow.
            return self.datadict.get(tag, None)
        return self.datadict.get(tag, default)

    @overload
    def param(self, tag: str, key: int | str) -> NumericType: ...

    @overload
    def param(self, tag: str, key: Sequence) -> pd.Series: ...

    def param(self, tag, key):
        """Retrieve the first value(s) of the specified column(s) from a tagged Data object.

        Parameters
        ----------
        col : str, list of str, or tuple of str
            Column name or list of column names.
        tag : str or None, optional
            Tag of the data group to retrieve.

        Returns
        -------
        NumericType or list of NumericType
            First value(s) in the column(s).

        """
        return self.get(tag).param(key)


# Local Variables:
# jinx-local-words: "StringIO csv datadict datadir dataframe datapath df noqa sep str"
# End:
