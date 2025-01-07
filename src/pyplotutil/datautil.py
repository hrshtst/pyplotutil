"""Data Handling and Manipulation Module.

This module provides classes for managing and manipulating tabular data, with functionalities to
load data from various sources, group data structure by specified tags, and access columns or rows
with intuitive syntax. The primary classes, `Data`, and `TaggedData`, facilitate working with
tabular data in polars DataFrame while allowing access to specific features like data grouping,
dynamic attribute setting, and easy retrieval of parameter values.

Classes
-------
BaseData : Abstract base class providing the core attributes and methods for data handling.
    Defines basic properties for data path and DataFrame storage.

Data : Extends BaseData to represent a single tabular data.
    Provides methods to access columns and retrieve specific parameters.

TaggedData : Extends BaseData to handle grouped data based on a specified tag column.
    Allows grouping data by a tag and accessing each group as a separate `Data` object.

Examples
--------
Basic usage:
    >>> data = Data("data.csv")
    >>> print(data.param("column_name"))

Tagged data usage:
    >>> tagged_data = TaggedData("data.csv", tag="tag")
    >>> data = tagged_data.get("specific_tag")
    >>> print(group.param("column_name"))

This module is designed to streamline operations with tabular data in data analysis, data plotting,
and other applications requiring structured data handling.

"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cached_property, partial
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, overload

import numpy as np
import polars as pl
from polars.exceptions import ColumnNotFoundError

from pyplotutil._typing import (
    FilePath,
    MultiColSelector,
    MultiIndexSelector,
    NoDefault,
    NumericType,
    SingleColSelector,
    SingleIndexSelector,
    Unknown,
    no_default,
)

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView

    from pyplotutil._typing import DataSourceType


_ERR_MSG_NOT_LOADED_FROM_FILE = "Data object may not be loaded from a file."
_ERR_MSG_PATH_NOT_EXIST = "Source path does not exist"
_WARN_MSG_CLONE_NOT_LOADED_FROM_FILE = "clone: Source Data object may not be loaded from a file."
_WARN_MSG_NO_FILES_FOUND = "No files found with the following glob pattern"


class BaseData:
    """Base class for data handling and manipulation.

    This class provides functionalities for setting and retrieving the path and the main DataFrame
    associated with the data.

    """

    _datapath: Path
    _dataframe: pl.DataFrame

    def __init__(
        self,
        data_source: DataSourceType,
        *,
        separator: str,
        has_header: bool,
        columns: Sequence[int] | Sequence[str] | None,
        names: Sequence[str] | None,
        n_rows: int | None,
        comment: str | None,
    ) -> None:
        """Initialize the BaseData object with the provided data source.

        Parameters
        ----------
        data_source : str | Path | StringIO | pl.DataFrame | pl.Series
            The data source.
        separator : str
            Single byte character to use as separator in the file.
        has_header : bool, optional
            Indicate if the first row of the dataset is a header or not.
        columns : Sequence[int], Sequence[str] or range, optional
            Columns to read from the data source.
        names : Sequence[str], optional
            Rename columns right after parsing the CSV file.
        n_rows : int, optional
            Number of rows to read.
        comment : str, optional
            Character to indicate comments in the data file.

        Raises
        ------
        TypeError
            If the data type is unsupported.

        """
        if isinstance(data_source, pl.DataFrame):
            self._set_dataframe(data_source)
        elif isinstance(data_source, pl.Series):
            self._set_dataframe(data_source.to_frame())
        elif isinstance(data_source, StringIO | FilePath):
            self._set_dataframe(
                self.read_csv(
                    data_source,
                    separator=separator,
                    has_header=has_header,
                    columns=columns,
                    names=names,
                    n_rows=n_rows,
                    comment=comment,
                ),
            )
            if isinstance(data_source, FilePath):
                self.set_datapath(data_source)
        else:
            msg = f"Unsupported data source type: {type(data_source)}"
            raise TypeError(msg)

    @staticmethod
    def read_commented_column_names(
        file_or_buffer: FilePath | StringIO,
        *,
        separator: str,
        comment: str,
    ) -> list[str] | None:
        """Return a list of column names extracted from commented lines in the file.

        Parameters
        ----------
        file_or_buffer : str | Path | StringIO
            The file or buffer containing the data.
        separator : str
            Single byte character to use as separator in the file..
        comment : str
            Character indicating commented lines.

        Returns
        -------
        list of str or None
            List of column names if found; otherwise, None.

        """

        def last_commented_header(buffer: TextIO, comment: str) -> str:
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
            n = len(comment)
            names = header[n:].strip().split(separator)
            if len(names) == 1 and names[0] == "":
                names = []
            return names
        return None

    @staticmethod
    def read_csv(
        file_or_buffer: FilePath | StringIO,
        *,
        separator: str,
        has_header: bool,
        columns: Sequence[int] | Sequence[str] | None,
        names: Sequence[str] | None,
        n_rows: int | None,
        comment: str | None,
    ) -> pl.DataFrame:
        """Return a polars DataFrame loaded from a file or string buffer.

        Parameters
        ----------
        file_or_buffer : str | Path | StringIO
            The file or buffer to read from.
        separator : str, optional
            Single byte character to use as separator in the file..
        has_header : bool, optional
            Indicate if the first row of the dataset is a header or not.
        columns : Sequence[int], Sequence[str] or range, optional
            Columns to read from the data source.
        names : Sequence[str], optional
            Rename columns right after parsing the CSV file.
        n_rows : int, optional
            Number of rows to read.
        comment : str, optional
            Character to indicate comments in the data file.

        Returns
        -------
        pl.DataFrame
            The loaded DataFrame.

        """
        if comment is not None and names is None:
            names = BaseData.read_commented_column_names(file_or_buffer, separator=separator, comment=comment)
            if names is not None:
                has_header = False
        if isinstance(file_or_buffer, StringIO):
            file_or_buffer.seek(0)
        return pl.read_csv(
            file_or_buffer,
            has_header=has_header,
            columns=columns,
            new_columns=names,
            separator=separator,
            comment_prefix=comment,
            n_rows=n_rows,
            rechunk=True,
        )

    def _set_dataframe(self, dataframe: pl.DataFrame) -> None:
        """Set the DataFrame associated with the data object.

        Parameters
        ----------
        dataframe : pl.DataFrame
            The DataFrame to associate with the data.

        """
        self._dataframe = dataframe

    @property
    def dataframe(self) -> pl.DataFrame:
        """Retrieve the raw DataFrame associated with the data.

        Returns
        -------
        pl.DataFrame
            The DataFrame associated with the data.

        """
        return self._dataframe

    @property
    def df(self) -> pl.DataFrame:
        """Alias for `dataframe` attribute.

        Returns
        -------
        pl.DataFrame
            The DataFrame associated with the data.

        """
        return self.dataframe

    def is_loaded_from_file(self) -> bool:
        """Check if the Data object was loaded from a file.

        Returns
        -------
        bool
            True if loaded from a file; otherwise, False.

        """
        try:
            _ = self._datapath
        except AttributeError:
            return False
        return True

    def set_datapath(self, datapath: str | Path) -> None:
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
        Path
            Path to the data file.

        """
        try:
            return self._datapath
        except AttributeError as e:
            msg = _ERR_MSG_NOT_LOADED_FROM_FILE
            raise AttributeError(msg) from e

    @property
    def datadir(self) -> Path:
        """Retrieve the directory of the data file.

        Returns
        -------
        Path
            Directory of the data file.

        """
        return self.datapath.parent

    def __str__(self) -> str:
        """Return a string of the associated DataFrame.

        Returns
        -------
        str
            String representation of the associated DataFrame object.

        """
        return str(self.dataframe)

    def __repr__(self) -> str:
        """Return a string representation of the object.

        Returns
        -------
        str
            String representation of the object.

        """
        if self.is_loaded_from_file():
            return f"{self.__class__.__name__}({self.datapath})"
        return f"{self.__class__.__name__}({self.dataframe})"


class Data(BaseData):
    """A class representing tabular data loaded from a source.

    This class provides methods for easy data access and specific data parameter extraction. When
    the source data contains column names, this class provides attributes for access the data column
    with its name.

    Attributes
    ----------
    dataframe : pl.DataFrame
        The DataFrame containing the tabular data.
    datapath : Path
        The path to the data file.
    datadir : Path
        The directory of the data file

    Examples
    --------
    Initialization of the data object and access for column data with its name.

    >>> import polars as pl
    >>> data = Data(pl.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": [5, 5, 5]}))
    >>> data.a
    0    1
    1    2
    2    3
    Name: a, dtype: int64
    >>> data.param("c")
    np.int64(5)

    """

    def __init__(
        self,
        data_source: DataSourceType,
        *,
        separator: str = ",",
        has_header: bool = True,
        columns: Sequence[int] | Sequence[str] | None = None,
        names: Sequence[str] | None = None,
        n_rows: int | None = None,
        comment: str | None = None,
    ) -> None:
        """Initialize the BaseData object with the provided data source.

        Parameters
        ----------
        data_source : str | Path | StringIO | pl.DataFrame | pl.Series
            The data source.
        separator : str
            Single byte character to use as separator in the file.
        has_header : bool, optional
            Indicate if the first row of the dataset is a header or not.
        columns : Sequence[int], Sequence[str] or range, optional
            Columns to read from the data source.
        names : Sequence[str], optional
            Rename columns right after parsing the CSV file.
        n_rows : int, optional
            Number of rows to read.
        comment : str, optional
            Character to indicate comments in the data file.

        Raises
        ------
        TypeError
            If the data type is unsupported.

        """
        super().__init__(
            data_source,
            separator=separator,
            has_header=has_header,
            columns=columns,
            names=names,
            n_rows=n_rows,
            comment=comment,
        )

    @overload
    def __getitem__(self, key: tuple[SingleIndexSelector, SingleColSelector]) -> Unknown: ...

    @overload
    def __getitem__(  # type: ignore[overload-overlap]
        self,
        key: str | tuple[MultiIndexSelector, SingleColSelector],
    ) -> pl.Series: ...

    @overload
    def __getitem__(
        self,
        key: (
            SingleIndexSelector
            | MultiIndexSelector
            | MultiColSelector
            | tuple[SingleIndexSelector, MultiColSelector]
            | tuple[MultiIndexSelector, MultiColSelector]
        ),
    ) -> pl.DataFrame: ...

    def __getitem__(
        self,
        key: (
            SingleIndexSelector
            | SingleColSelector
            | MultiColSelector
            | MultiIndexSelector
            | tuple[SingleIndexSelector, SingleColSelector]
            | tuple[SingleIndexSelector, MultiColSelector]
            | tuple[MultiIndexSelector, SingleColSelector]
            | tuple[MultiIndexSelector, MultiColSelector]
        ),
    ) -> pl.DataFrame | pl.Series | Unknown:
        """Access a specific column(s).

        Parameters
        ----------
        key : str or int or Sequence of str or int
            Column name or column index.

        Returns
        -------
        pl.Series or pl.DataFrame
            Series or frame of the specified column(s).

        """
        return self.dataframe.__getitem__(key)

    def __len__(self) -> int:
        """Return the number of rows in the `Data` object.

        Returns
        -------
        int
            Number of rows in the `Data` object.

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
        Unknown
            The attribute from the DataFrame.

        """
        if name in ("datapath", "datadir"):
            return self.__getattribute__(name)
        if name in self.dataframe.columns:
            return self.dataframe.get_column(name)
        return getattr(self.dataframe, name)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Return an iterator over the Data objects.

        Returns
        -------
        Iterator[np.ndarray]
            An iterator over the Data objects.

        """
        return iter(self.dataframe.to_numpy())

    def split_by_row(self, row_index: int) -> tuple[Data, Data]:
        """Split the Data object into two parts at a specified row index.

        Parameters
        ----------
        row_index : int
            The index at which to split the data object. Rows from the start up to
            `row_index` will go to the first split, and rows from `row_index` to
            the end will go to the second split.

        Returns
        -------
        tuple[Data, Data]
            A tuple containing two Data objects. The first contains rows from the
            start to `row_index`, and the second contains rows from `row_index`
            to the end.

        """
        df1 = self.dataframe[:row_index]
        df2 = self.dataframe[row_index:]
        return Data(df1), Data(df2)

    @overload
    def param(self, key: int | str) -> NumericType: ...

    @overload
    def param(self, key: Sequence) -> tuple[NumericType, ...]: ...

    def param(self, key):
        """Retrieve specific parameter(s) for column(s).

        Parameters
        ----------
        key : int or str or Sequence of int or str
            The column(s) for which to retrieve the parameter.

        Returns
        -------
        Numeric type or pl.Series
            Retrieved parameter value(s).

        """
        subset = self.dataframe[:, key]
        if isinstance(subset, pl.Series):
            return subset[0]
        return subset.row(0)

    def _copy_datapath_if_loaded_from_file(self, dest_data: Data) -> Data:
        try:
            dest_data.set_datapath(self.datapath)
        except AttributeError as e:
            if _ERR_MSG_NOT_LOADED_FROM_FILE in str(e):
                msg = _WARN_MSG_CLONE_NOT_LOADED_FROM_FILE
                warnings.warn(msg, UserWarning, stacklevel=2)
            else:
                raise AttributeError(str(e)) from e
        return dest_data

    def clone(
        self,
        *,
        rename_mapping: dict[str, str] | Sequence[str] | None = None,
        keep_datapath: bool = False,
    ) -> Data:
        if isinstance(rename_mapping, Mapping):
            cloned_df = self.df.rename(rename_mapping)
        elif isinstance(rename_mapping, Sequence):
            cloned_df = self.df.rename(dict(zip(self.df.columns, rename_mapping, strict=True)))
        else:
            cloned_df = self.df.clone()
        cloned_data = Data(cloned_df)
        if keep_datapath:
            cloned_data = self._copy_datapath_if_loaded_from_file(cloned_data)
        return cloned_data

    def subset(
        self,
        key: Sequence[str],
        *,
        start: int | None = None,
        end: int | None = None,
        rename_mapping: dict[str, str] | Sequence[str] | None = None,
        keep_datapath: bool = False,
    ) -> Data:
        subset_df = self[start:end, key]
        if isinstance(rename_mapping, Mapping):
            subset_df = subset_df.rename(rename_mapping)
        elif isinstance(rename_mapping, Sequence):
            subset_df = subset_df.rename(dict(zip(subset_df.columns, rename_mapping, strict=True)))
        subset_data = Data(subset_df)
        if keep_datapath:
            subset_data = self._copy_datapath_if_loaded_from_file(subset_data)
        return subset_data


class TaggedData(BaseData):
    """A class for handling data grouped by a specified tag.

    This class provides methods to load and access data grouped by a tag, enabling
    easy access to each group's data as individual `Data` objects.

    Attributes
    ----------
    dataframe : pl.DataFrame
        The DataFrame containing the tabular data with tags.
    datadict : dict of str to Data
        A dictionary mapping each tag value to a corresponding `Data` object.

    """

    _datadict: dict[str, Data]
    _tag_column_name: str

    def __init__(
        self,
        data_source: DataSourceType,
        *,
        separator: str = ",",
        has_header: bool = True,
        columns: Sequence[int] | Sequence[str] | None = None,
        names: Sequence[str] | None = None,
        n_rows: int | None = None,
        comment: str | None = None,
        tag_column: str = "tag",
    ) -> None:
        """Initialize the BaseData object with the provided data source.

        Parameters
        ----------
        data_source : str | Path | StringIO | pl.DataFrame | pl.Series
            The data source.
        separator : str
            Single byte character to use as separator in the file.
        has_header : bool, optional
            Indicate if the first row of the dataset is a header or not.
        columns : Sequence[int], Sequence[str] or range, optional
            Columns to read from the data source.
        names : Sequence[str], optional
            Rename columns right after parsing the CSV file.
        n_rows : int, optional
            Number of rows to read.
        comment : str, optional
            Character to indicate comments in the data file.
        tag_column : str, optional
            Column name used to tag and group data.

        Raises
        ------
        TypeError
            If the data type is unsupported.

        """
        super().__init__(
            data_source,
            separator=separator,
            has_header=has_header,
            columns=columns,
            names=names,
            n_rows=n_rows,
            comment=comment,
        )
        self._tag_column_name = tag_column
        self._make_groups(self._tag_column_name)

    def __iter__(self) -> Iterator[tuple[str, Data]]:
        """Return an iterator over the grouped Data objects.

        Each group is represented by a tuple of (tag, Data).

        Returns
        -------
        Iterator[tuple[str, Data]]
            An iterator over the grouped Data objects.

        """
        return iter(self.datadict.items())

    def __len__(self) -> int:
        """Return the number of Data objects divided by the tag.

        Returns
        -------
        int
            Number of Data objects divided by the tag.

        """
        return len(self._datadict)

    def _make_groups(self, tag_column_name: str) -> None:
        """Group the data by the specified tag and stores it in `datadict`."""
        try:
            self._datadict = {
                str(name[0]): Data(group.drop(tag_column_name))
                for name, group in self.dataframe.group_by(tag_column_name)
            }
        except ColumnNotFoundError:
            self._datadict = {"unknown": Data(self.dataframe)}

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
        KeysView of str
            Dictionary keys of tags.

        """
        return self.datadict.keys()

    def items(self) -> ItemsView[str, Data]:
        """Retrieve the items (tag and Data object) of the grouped data.

        Returns
        -------
        ItemsView of tuple of (str, Data)
            Dictionary items of tag-Data object pairs.

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
        tag : str
            Tag of the data group to retrieve.

        default : Data or None, optional
            The default data object when the specified tag is not found.

        Returns
        -------
        Data
            Data object corresponding to the tag.

        Raises
        ------
        KeyError
            If the specified tag value does not exist and no default value is given.

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
    def param(self, tag: str, key: Sequence) -> tuple[NumericType, ...]: ...

    def param(self, tag, key):
        """Retrieve specific parameter(s) for column(s) from a tagged Data object.

        Parameters
        ----------
        tag : str
            Tag of the data group to retrieve.

        key : int or str or Sequence of int or str
            The column(s) for which to compute the parameter.

        Returns
        -------
        Numeric type or pl.Series
            Computed parameter value(s).

        """
        return self.get(tag).param(key)

    def __str__(self) -> str:
        """Return a string of the grouped mapping of tag to Data.

        Returns
        -------
        str
            String representation of the grouped mapping of tag to Data.

        """
        return str(self.datadict)

    def __repr__(self) -> str:
        """Return a string representation of the object.

        Returns
        -------
        str
            String representation of the object.

        """
        if self.is_loaded_from_file():
            return f"{self.__class__.__name__}({self.datapath}, tag={self._tag_column_name})"
        return f"{self.__class__.__name__}({self.dataframe}, tag={self._tag_column_name})"


class Dataset:
    _dataset: list[Data]

    def __init__(
        self,
        source_paths: FilePath | Iterable[FilePath],
        *,
        separator: str = ",",
        has_header: bool = True,
        columns: Sequence[int] | Sequence[str] | None = None,
        names: Sequence[str] | None = None,
        n_rows: int | None = None,
        comment: str | None = None,
        glob_pattern: str = "**/*.csv",
        n_pickup_per_directory: int | None = None,
    ) -> None:
        self._dataset = Dataset.load_dataset(
            source_paths,
            separator=separator,
            has_header=has_header,
            columns=columns,
            names=names,
            n_rows=n_rows,
            comment=comment,
            glob_pattern=glob_pattern,
            n_pickup_per_directory=n_pickup_per_directory,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[Data]:
        return iter(self._dataset)

    @overload
    def __gettiem__(self, index: int) -> Data: ...

    @overload
    def __gettiem__(self, index: slice) -> list[Data]: ...

    def __gettiem__(self, index):
        return self._dataset[index]

    @property
    def datapaths(self) -> list[Path]:
        return [data.datapath for data in self]

    @property
    def datadirs(self) -> list[Path]:
        return list(OrderedDict.fromkeys(self.datapaths))

    @property
    def dataset(self) -> list[Data]:
        return self._dataset

    @staticmethod
    def load_dataset(
        source_paths: FilePath | Iterable[FilePath],
        *,
        separator: str = ",",
        has_header: bool = True,
        columns: Sequence[int] | Sequence[str] | None = None,
        names: Sequence[str] | None = None,
        n_rows: int | None = None,
        comment: str | None = None,
        glob_pattern: str = "**/*.csv",
        n_pickup_per_directory: int | None = None,
    ) -> list[Data]:
        if isinstance(source_paths, FilePath):
            source_paths = [source_paths]

        data_loader: Callable[[Path], Data] = partial(
            Data,
            separator=separator,
            has_header=has_header,
            columns=columns,
            names=names,
            n_rows=n_rows,
            comment=comment,
        )

        dataset: list[Data] = []
        for source_path in source_paths:
            path = Path(source_path)
            if not path.exists():
                msg = f"{_ERR_MSG_PATH_NOT_EXIST}: {path!s}"
                raise ValueError(msg)
            if path.is_dir():
                found_files = sorted(path.glob(glob_pattern))
                if n_pickup_per_directory is not None:
                    found_files = found_files[:n_pickup_per_directory]
                if len(found_files) == 0:
                    msg = f"{_WARN_MSG_NO_FILES_FOUND}: {path!s}, {glob_pattern}"
                    warnings.warn(msg, UserWarning, stacklevel=2)
                dataset.extend(data_loader(x) for x in found_files)
            else:
                dataset.append(data_loader(path))
        return dataset

    @cached_property
    def min_n_rows(self) -> int:
        return min(map(len, self._dataset))

    def get_columns(self, name: str, *, aligned: bool = True) -> list[pl.Series]:
        if not aligned:
            return [data.get_column(name) for data in self]
        n = self.min_n_rows
        return [data.get_column(name)[:n] for data in self]

    def get_columns_as_array(self, name: str) -> np.ndarray:
        return np.asarray(self.get_columns(name, aligned=True))

    def get_axis_data(
        self,
        x_axis_name: str,
        y_data_name: str,
        *,
        x_axis_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        y = self.get_columns_as_array(y_data_name)
        n = y.shape[1]
        x = self._dataset[x_axis_index].get_column(x_axis_name).to_numpy()[:n]
        return x, y

    def get_timeseries(
        self,
        name: str,
        t_shift: float = 0.0,
        *,
        t_axis_name: str = "t",
        t_axis_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        t, y = self.get_axis_data(t_axis_name, name, x_axis_index=t_axis_index)
        t = t - t_shift
        return t, y


# Local Variables:
# jinx-local-words: "StringIO csv datadict datadir dataframe datapath dataset dtype ndarray np param polars str"
# End:
