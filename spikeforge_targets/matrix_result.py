"""The typed, JSON-serialisable result of a capability classification."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from spikeforge_targets.substitution import Substitution


@dataclass(frozen=True)
class CapabilityMatrix:
    """Per-node classification of one graph against one target.

    Every node of the classified graph appears in exactly one of
    ``supported``, ``unsupported``, or ``substituted``; the buckets therefore
    partition the node set and no node can be silently dropped.
    """

    target: str
    available: bool
    supported: Tuple[str, ...]
    unsupported: Tuple[str, ...]
    substituted: Tuple[Substitution, ...]

    def counts(self) -> Dict[str, int]:
        """Return the node count of every bucket plus the total."""
        return {
            "supported": len(self.supported),
            "unsupported": len(self.unsupported),
            "substituted": len(self.substituted),
            "total": (
                len(self.supported)
                + len(self.unsupported)
                + len(self.substituted)
            ),
        }

    def deployable(self) -> bool:
        """Return True when the target is available and has no gap."""
        return self.available and not self.unsupported

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the classification."""
        return {
            "target": self.target,
            "available": self.available,
            "supported": list(self.supported),
            "unsupported": list(self.unsupported),
            "substituted": [item.to_dict() for item in self.substituted],
            "counts": self.counts(),
            "deployable": self.deployable(),
        }
