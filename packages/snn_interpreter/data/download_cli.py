"""Download a single dataset in an isolated child process.

Run as ``python -m snn_interpreter.data.download_cli <dataset> <train>`` so
the parent server can terminate the process to cancel an in-flight download
while keeping the FastAPI event loop responsive.
"""

import sys
from typing import List

from snn_interpreter.data.datasets import build_dataset, dataset_modality
from snn_interpreter.data.event_loader import ensure_event_dataset


def _ensure(dataset: str, train: bool) -> None:
    """Prepare one dataset, routing event modality to its tonic loader."""
    if dataset_modality(dataset) == "event":
        ensure_event_dataset(dataset)
    else:
        build_dataset(dataset, train=train, download=True)


def main(argv: List[str]) -> int:
    """Download the dataset named on the command line, then exit."""
    _ensure(argv[1], argv[2] == "1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
