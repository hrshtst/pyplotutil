# pyplotutil: plotting utility for academic publications

**pyplotutil** is a wrapper library of
[polars](https://pola.rs/) and
[matplotlib](https://matplotlib.org/), which allows you to handle time
series data easily and create graphs tolerable to scientific
publications.

## Features

- Load tabular data from CSV, Parquet, JSON/NDJSON, or Excel files,
  buffers, or polars frames with column-attribute access
  (`data.time`, `data.voltage`)
- Group rows by a tag column (`TaggedData`, `load_tagged_data`) or
  manage whole directories of data files (`Dataset`)
- Plot multiple time series and mean-with-error graphs (standard
  deviation, standard error, variance, range, confidence interval)
- Plot directly from data objects: one line per file for a
  `Dataset`, one labeled line per tag for `TaggedData`
- Apply publication-ready matplotlib styles
  ([SciencePlots](https://github.com/garrettj403/SciencePlots)):
  `science`, `ieee`, `nature`, `notebook`
- Annotate plotted lines with boxed labels and arrows
  (`annotate_with_arrow`) and add direction arrowheads along
  trajectories (`add_direction_arrows`)
- Highlight time spans from boolean masks (`shade_spans`) and
  configure axes cosmetics in one call (`setup_axes`)
- Save figures to multiple formats in one call with sanitized
  filenames
- Event logging helpers with file and console output

## Installation

Requires Python 3.11+.

```console
pip install pyplotutil
```

The latest development version can be installed directly from GitHub:

```console
pip install git+https://github.com/hrshtst/pyplotutil.git
```

Interactive plotting with `plt.show()` needs a GUI backend, which is
provided by the optional `gui` extra:

```console
pip install "pyplotutil[gui]"
```

## Quickstart

```python
import matplotlib.pyplot as plt

from pyplotutil import Data, Dataset, apply_style, plot_mean_err, save_figure

# Publication-ready style.
apply_style("science", no_latex=True)

# Load a single CSV file; columns are accessible as attributes.
data = Data("experiment.csv")
fig, ax = plt.subplots()
ax.plot(data.t, data.position)

# Load every CSV file in a directory and plot mean with standard error.
dataset = Dataset("results/")
plot_mean_err(ax, dataset, "position", "se", capsize=2, label="mean")

# Write figure.png and figure.pdf in one call.
save_figure(fig, "output", "figure", ["png", "pdf"])
```

Group rows by a tag column:

```python
from pyplotutil import TaggedData

tagged = TaggedData("trials.csv", tag_column="trial")
for tag, data in tagged:
    print(tag, data.param("gain"))
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) and
[nox](https://nox.thea.codes/):

```console
uv sync                 # set up the development environment
uv run pytest           # run the test suite
uv run mypy             # type check
uv run ruff check       # lint
nox                     # lint + type check + tests on Python 3.11-3.14
```

## License

[MIT](LICENSE)
