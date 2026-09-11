"""Typed errors raised by the energy/latency accounting package.

Grouping them here mirrors :mod:`spikeforge_hub.errors`: a malformed
declared cost table is a named failure rather than a bare traceback, so a bad
bundled table is reported by target instead of silently ignored.
"""


class EnergyError(Exception):
    """Base class for every typed energy-accounting failure."""


class EnergyCostError(EnergyError):
    """Raised when a declared cost table cannot be read or validated.

    The offending ``target`` (the table's file stem) and a human ``detail``
    are stored as attributes so a caller can name which table is unusable.
    """

    def __init__(self, target: str, detail: str) -> None:
        """Record ``target`` and ``detail`` and build a clear message."""
        super().__init__(f"invalid cost table for {target!r}: {detail}")
        self.target: str = target
        self.detail: str = detail
