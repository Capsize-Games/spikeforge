"""Download a single dataset in an isolated child process.

Run as ``python -m spikeforge.data.download_cli <dataset> <train>`` so
the parent server can terminate the process to cancel an in-flight download
while keeping the FastAPI event loop responsive.
"""

import sys
from typing import List

from spikeforge.data.datasets import (
    build_dataset,
    dataset_modality,
    dataset_splits,
)
from spikeforge.data.event_loader import ensure_event_dataset


def _ensure(dataset: str, train: bool) -> None:
    """Prepare one dataset, routing event modality to its tonic loader.

    Every declared split of an event dataset is warmed, not just the
    training one: held-out scoring runs inside the training process, so a
    cold test split would pull from the network there rather than here in
    the cancellable child. Image datasets keep taking their split from the
    caller's ``train`` flag.
    """
    if dataset_modality(dataset) == "event":
        for split in dataset_splits(dataset):
            ensure_event_dataset(dataset, split=split)
    else:
        build_dataset(dataset, train=train, download=True)


def main(argv: List[str]) -> int:
    """Download the dataset named on the command line, then exit."""
    _ensure(argv[1], argv[2] == "1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
