#!/usr/bin/env python

"""Generate sample data used in example scripts."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, TypeVar

import numpy as np
import polars as pl

T = TypeVar("T", float, np.ndarray)

this_script_path = Path(__file__).resolve()
data_dir_path = this_script_path.parent / "data"
tagged_data_dir_path = this_script_path.parent / "tagged_data"

for dir_path in (data_dir_path, tagged_data_dir_path):
    dir_path.mkdir(parents=True, exist_ok=True)


def model(x: T, p: int) -> T:
    """Generate sample data used in example scripts."""
    return x ** (2 * p + 1) / (1 + x ** (2 * p))


x = np.linspace(0.75, 1.25, 201)
tagged_y: list[tuple[str, np.ndarray]] = []
for p in [10, 15, 20, 30, 50, 100]:
    y = model(x, p)
    pl.DataFrame({"x": x, "y": y, "p": [p] * len(x)}).write_csv(data_dir_path / f"model_{p:03d}.csv")
    tagged_y.append((f"p{p:03d}", y))


class DATASET(TypedDict):
    """TypedDict for tagged data object."""

    tag: list[str]
    x: list[float]
    y: list[float]
    p: list[int]


dataset: DATASET = {"tag": [], "x": [], "y": [], "p": []}
for tag, y in tagged_y:
    dataset["tag"].extend([tag] * len(y))
    dataset["x"].extend(x)
    dataset["y"].extend(y)
    dataset["p"].extend([int(tag[1:])] * len(y))
pl.DataFrame(dataset).write_csv(tagged_data_dir_path / "models.csv")

# Local Variables:
# jinx-local-words: "csv env usr"
# End:
