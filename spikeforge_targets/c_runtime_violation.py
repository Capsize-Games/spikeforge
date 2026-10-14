"""One export node the C runtime cannot load, with its reason."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class CRuntimeViolation:
    """A single ``nir-c-runtime`` incompatibility.

    ``name`` is the node (or ``source->target`` edge, or ``graph`` for a
    graph-level check), ``kind`` its NIR node kind, and ``reason`` the
    exact C-loader assumption it breaks.
    """

    name: str
    kind: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the violation."""
        return {"name": self.name, "kind": self.kind, "reason": self.reason}
