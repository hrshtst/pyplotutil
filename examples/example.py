from __future__ import annotations

import argparse

from pyplotutil.datautil import Data
from pyplotutil.loggingutil import evlog, start_logging


def parse() -> argparse.Namespace:
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
    import sys

    args = parse()
    evlog().debug("This debug message is emitted before starting logging")
    start_logging(sys.argv, args.output_dir, __name__, args.verbose)
    evlog().info("Output directory: %s", args.output_dir)
    evlog().debug("Parsed arguments: %s", args)

    data = Data(args.data)
    evlog().info("Loaded data file: %s", args.data)

    evlog().warning("This is WARNING message")
    evlog().error("This is ERROR message")
    evlog().critical("This is CRITICAL message")


if __name__ == "__main__":
    main()
