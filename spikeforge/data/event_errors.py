"""Errors raised by the event dataset loader."""


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
