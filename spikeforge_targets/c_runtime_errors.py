"""Typed refusal for exports the C runtime cannot load."""

from typing import Sequence, Tuple

from spikeforge_targets.c_runtime_violation import CRuntimeViolation


class CRuntimeUnsupportedError(Exception):
    """Raised when an export violates the ``nir-c-runtime`` profile.

    The offending ``violations`` are stored as an attribute so callers
    can react without parsing the message; ``names`` and ``kinds`` name
    the incompatible nodes directly.
    """

    def __init__(self, violations: Sequence[CRuntimeViolation]) -> None:
        """Record ``violations`` and build a message naming each one."""
        items = tuple(violations)
        details = "; ".join(
            f"{item.name!r} ({item.kind}): {item.reason}" for item in items
        )
        super().__init__(f"export cannot load in nir-c-runtime: {details}")
        self.violations: Tuple[CRuntimeViolation, ...] = items
        self.names: Tuple[str, ...] = tuple(item.name for item in items)
        self.kinds: Tuple[str, ...] = tuple(item.kind for item in items)
