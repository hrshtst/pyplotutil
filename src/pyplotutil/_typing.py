from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, auto
from io import StringIO
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, TypeVar

import numpy as np
import polars as pl

Unknown: TypeAlias = Any
FilePath: TypeAlias = str | Path
DataSourceType: TypeAlias = FilePath | StringIO | pl.DataFrame | pl.Series
NumericType: TypeAlias = int | float | complex | np.number
NumericTypeVar = TypeVar("NumericTypeVar", bound=NumericType)

SingleIndexSelector: TypeAlias = int
MultiIndexSelector: TypeAlias = slice | range | Sequence[int] | pl.Series | np.ndarray
SingleNameSelector: TypeAlias = str
MultiNameSelector: TypeAlias = slice | Sequence[str] | pl.Series | np.ndarray
BooleanMask: TypeAlias = Sequence[bool] | pl.Series | np.ndarray
SingleColSelector: TypeAlias = SingleIndexSelector | SingleNameSelector
MultiColSelector: TypeAlias = MultiIndexSelector | MultiNameSelector | BooleanMask


class _NoDefault(Enum):
    """Enum to represent the absence of a default value in method parameters."""

    no_default = auto()


no_default: Final = _NoDefault.no_default
NoDefault: TypeAlias = Literal[_NoDefault.no_default]

# Local Variables:
# jinx-local-words: "Enum"
# End:
