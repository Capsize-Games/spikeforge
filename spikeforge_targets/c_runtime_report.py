"""The typed, JSON-serialisable result of a C-runtime validation."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from spikeforge_targets.c_runtime_violation import CRuntimeViolation


@dataclass(frozen=True)
class CRuntimeReport:
    """Per-node verdict of one export against the C runtime profile.

    ``supported`` names the nodes the C loader implements with conforming
    parameters; ``unsupported`` carries one violation per incompatible
    node, edge, or graph-level check. ``passed`` is True only when every
    node, edge, and structural assumption conforms.
    """

    target: str
    supported: Tuple[str, ...]
    unsupported: Tuple[CRuntimeViolation, ...]

    def passed(self) -> bool:
        """Return True when the export conforms to the C profile."""
        return not self.unsupported

    def counts(self) -> Dict[str, int]:
        """Return the supported/unsupported node counts plus the total."""
        return {
            "supported": len(self.supported),
            "unsupported": len(self.unsupported),
            "total": len(self.supported) + len(self.unsupported),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the report."""
        return {
            "target": self.target,
            "supported": list(self.supported),
            "unsupported": [item.to_dict() for item in self.unsupported],
            "counts": self.counts(),
            "passed": self.passed(),
        }
