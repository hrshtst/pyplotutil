"""Example script of pyplotutil module."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from pyplotutil.datautil import Data
from pyplotutil.loggingutil import evlog, start_logging
from pyplotutil.plotutil import apply_ieee_style


def parse() -> argparse.Namespace:
    """Parse provided command arguments."""
    parser = argparse.ArgumentParser(description="Compare scores across preview/delay steps")
    parser.add_argument("data", help="CSV file.")
    parser.add_argument("-o", "--output-directory", dest="output_dir", default=".", help="Output directory")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose output. -v provides additional info. -vv provides debug output.",
    )
    return parser.parse_args()


def main() -> None:
    """Run main function."""
    import sys

    args = parse()
    evlog().debug("This debug message is emitted before starting logging")
    start_logging(sys.argv, args.output_dir, __name__, args.verbose)
    evlog().info("Output directory: %s", args.output_dir)
    evlog().debug("Parsed arguments: %s", args)

    data = Data(args.data)
    evlog().info("Loaded data file: %s", args.data)

    apply_ieee_style()
    plt.plot(data[:, 0], data[:, 1])
    plt.show()


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "pyplotutil"
# End:
