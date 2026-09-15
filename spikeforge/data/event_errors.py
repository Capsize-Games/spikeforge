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


class EventTimestampError(ValueError):
    """Raised when a stream's timestamps cannot be real recording times.

    Microsecond timestamps are non-negative, so a negative one is not data:
    it is the signature of a non-finite float cast to an integer, and
    ``int(float("nan"))`` under numpy lands on ``INT64_MIN``.

    This is not hypothetical. Tonic's SHD/SSC reader scales the file's
    timestamps by ``1e6`` to convert seconds to microseconds, but the
    Heidelberg files store them as ``float16``, whose maximum is 65504 -- so
    the multiply overflows to ``inf``, becomes ``NaN``, and every timestamp
    in the sample casts to ``INT64_MIN``. Binning then sees a zero-width
    time span and collapses every event into the first time step, which
    trains and scores perfectly happily while having destroyed all timing.
    A named failure is the only honest outcome.
    """

    def __init__(self, detail: str = "") -> None:
        """Explain what was seen and why it cannot be a recording time."""
        super().__init__(
            "event timestamps are negative, so they are not recording times "
            f"{detail}; this is what a non-finite timestamp cast to an "
            "integer looks like. Tonic's SHD/SSC reader produces it by "
            "scaling float16 seconds by 1e6 (the multiply overflows to inf, "
            "then NaN, then INT64_MIN), which would silently collapse every "
            "event into one time bin. Refusing rather than reporting a "
            "number measured on destroyed timing."
        )
