"""Typed errors raised by the sparse/event-driven runtime.

Grouping them here mirrors :mod:`spikeforge_hub.errors`: a failure a
caller may want to react to is a named type, so an unsupported stage kind is
reported rather than silently falling back to the dense path.
"""


class SparseError(Exception):
    """Base class for every typed sparse-runtime failure."""


class UnsupportedKindError(SparseError):
    """Raised when a stage kind has no event-driven implementation.

    The offending ``kind`` and ``stage`` are stored as attributes so a caller
    can name exactly which stage forced the failure instead of seeing an
    opaque traceback; the runtime never degrades to a silent dense fallback.
    """

    def __init__(self, kind: str, stage: str = "") -> None:
        """Record ``kind`` and ``stage`` and build a clear message."""
        detail = f" at stage {stage!r}" if stage else ""
        super().__init__(
            "sparse runtime has no event-driven implementation for kind "
            f"{kind!r}{detail}"
        )
        self.kind: str = kind
        self.stage: str = stage
