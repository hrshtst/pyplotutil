from __future__ import annotations

from enum import Enum, auto
from io import StringIO
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, TypeVar

import numpy as np
import pandas as pd

Unknown: TypeAlias = Any
FilePath: TypeAlias = str | Path
DataSourceType: TypeAlias = FilePath | StringIO | pd.DataFrame | pd.Series
NumericType: TypeAlias = int | float | complex | np.number
NumericTypeVar = TypeVar("NumericTypeVar", bound=NumericType)


class _NoDefault(Enum):
    """Enum to represent the absence of a default value in method parameters."""

    no_default = auto()


no_default: Final = _NoDefault.no_default
NoDefault: TypeAlias = Literal[_NoDefault.no_default]

# Local Variables:
# jinx-local-words: "Enum"
# End:
