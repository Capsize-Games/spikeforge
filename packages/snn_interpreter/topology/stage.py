"""A single declarative stage within a topology graph."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class Stage:
    """One named node in a topology graph.

    ``kind`` selects how the stage is rendered (``linear``, ``conv2d``,
    ``flatten``, ``avgpool2d``, ``sumpool2d``, ``add``, or a neuron kind),
    while ``params`` carries the constructor arguments for that kind.
    """

    name: str
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-able representation of this stage."""
        return {
            "name": self.name,
            "kind": self.kind,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Stage":
        """Rebuild a stage from :meth:`to_dict` output."""
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            params=dict(data.get("params", {})),
        )
