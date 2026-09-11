"""The interface every external tracking sink implements.

A sink has one responsibility: forward a manifest-like record to an external
tracker. It is deliberately tiny so the local, file-based manifest stays the
default and a sink is only ever an additional, opt-in destination.
"""

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Sink(Protocol):
    """A destination that receives a manifest-like tracking record."""

    @property
    def name(self) -> str:
        """Return the sink's stable identifier (``tensorboard``/``wandb``)."""
        ...

    @property
    def reason(self) -> str:
        """Return a human-readable note on availability."""
        ...

    def available(self) -> bool:
        """Return True when the sink's backend can actually be reached."""
        ...

    def log(self, record: Mapping[str, Any]) -> bool:
        """Forward ``record``; return True when it was delivered."""
        ...
