"""Errors raised by the event dataset loader."""

from typing import Sequence


class EventsExtraMissingError(RuntimeError):
    """Raised when an event dataset is used without the ``events`` extra.

    The message names the missing extra and the install command so callers
    never have to guess why an event dataset is unavailable, and so the
    failure surfaces as a typed error rather than a bare ``ImportError``.
    """

    def __init__(self, dataset: str = "") -> None:
        """Name the offending dataset and the extra that provides it."""
        detail = f" for dataset {dataset!r}" if dataset else ""
        super().__init__(
            "tonic is not installed, so event datasets are unavailable"
            f"{detail}; install the `events` extra with "
            'pip install -e ".[events]"'
        )


class EventSplitMissingError(RuntimeError):
    """Raised when a dataset declares no arguments for a requested split.

    Tonic's event datasets disagree on how a split is selected, so the
    registry records the arguments per dataset. A dataset that declares no
    entry for the requested split ships no such partition — CIFAR10-DVS
    arrives as one undivided pool — and asking for it raises this instead of
    quietly serving another split, which is how training data ends up
    reported as a held-out score.
    """

    def __init__(
        self, dataset: str, split: str, declared: Sequence[str] = ()
    ) -> None:
        """Name the dataset, the split asked for, and the ones it has."""
        names = ", ".join(declared) if declared else "none"
        super().__init__(
            f"dataset {dataset!r} declares no {split!r} split (it declares: "
            f"{names}), so it cannot be scored on held-out data; train on a "
            "dataset that ships a test split, or declare a deterministic "
            "partition for it in the registry"
        )
