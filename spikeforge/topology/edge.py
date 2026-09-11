"""A directed edge between two stages in a topology graph."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class Edge:
    """A directed link between stages.

    ``delayed`` marks a one-step feedback edge: the target reads the
    source's output from the *previous* time step. Delayed edges are
    excluded from cycle detection so recurrent graphs stay valid.
    """

    source: str
    target: str
    delayed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-able representation of this edge."""
        return {
            "source": self.source,
            "target": self.target,
            "delayed": self.delayed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Edge":
        """Rebuild an edge from :meth:`to_dict` output."""
        return cls(
            source=str(data["source"]),
            target=str(data["target"]),
            delayed=bool(data.get("delayed", False)),
        )
