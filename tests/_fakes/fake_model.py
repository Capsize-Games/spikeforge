"""The shared stand-in for the active training engine.

Four server-action test files carried their own ``FakeModel``; this is
their union. ``mode`` defaults to ``educational`` -- the mode the
server assumes for a double without one -- and the mode tests keep
passing theirs explicitly.
"""

from typing import Any


class FakeModel:
    """A minimal stand-in for the active training engine."""

    def __init__(self, spec: Any, net: Any, mode: str = "educational") -> None:
        """Store the topology spec, module, and execution mode."""
        self.spec = spec
        self.net = net
        self.mode = mode
