"""Download a single dataset in an isolated child process.

Run as ``python -m snn_interpreter.data.download_cli <dataset> <train>`` so
the parent server can terminate the process to cancel an in-flight download
while keeping the FastAPI event loop responsive.
"""

import sys
from typing import List

from snn_interpreter.data.datasets import build_dataset


def main(argv: List[str]) -> int:
    """Download the dataset named on the command line, then exit."""
    dataset = argv[1]
    train = argv[2] == "1"
    build_dataset(dataset, train=train, download=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
