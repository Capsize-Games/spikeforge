"""Typed errors raised by the ``spikeforge-io`` adapters.

Grouped here (mirroring :mod:`spikeforge_hub.errors` and
:mod:`spikeforge.serving.errors`) so a caller can react to a missing file, a
malformed row, a wrong channel count, or an absent optional dependency without
parsing a message. An adapter never silently reshapes bad input: a refusal is a
named type.
"""

from typing import Optional


class IoError(Exception):
    """Base class for every typed I/O-adapter failure."""


class AdapterError(IoError):
    """Base error for reading one recorded source.

    The offending ``source`` (a path, a dataset name, or ``"<memory>"``) and a
    human ``detail`` are stored as attributes.
    """

    def __init__(self, source: str, detail: str) -> None:
        """Record ``source`` and ``detail`` and build a clear message."""
        super().__init__(f"I/O adapter failed for {source!r}: {detail}")
        self.source: str = source
        self.detail: str = detail


class AdapterFormatError(AdapterError):
    """Raised when a source cannot be parsed into a numeric stream."""

    def __init__(
        self, source: str, detail: str, position: Optional[int] = None
    ) -> None:
        """Record the malformed ``detail`` and an optional ``position``."""
        super().__init__(source, detail)
        self.position: Optional[int] = position


class AdapterShapeError(AdapterError):
    """Raised when a source's shape is not a usable ``[N, D]`` stream."""

    def __init__(self, source: str, detail: str) -> None:
        """Record the shape ``detail``."""
        super().__init__(source, detail)


class AdapterUnsupportedError(AdapterError):
    """Raised when no adapter is registered for a source's suffix."""

    def __init__(self, source: str, detail: str) -> None:
        """Record why the source has no adapter."""
        super().__init__(source, detail)


class AdapterDependencyError(IoError):
    """Raised when an optional dependency (for example ``numpy``) is absent.

    The message names the dependency and how to install it, so an unavailable
    format is reported rather than guessed at.
    """

    def __init__(self, dependency: str, feature: str) -> None:
        """Name the missing dependency and the feature that needs it."""
        super().__init__(
            f"{dependency} is required to read {feature}; install the "
            "`spikeforge-io[npy]` extra"
        )
        self.dependency: str = dependency
        self.feature: str = feature
