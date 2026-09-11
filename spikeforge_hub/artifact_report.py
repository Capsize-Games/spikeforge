"""The structural report produced by artifact inspection.

Kept separate from the readers so a report can be built and compared by
:mod:`spikeforge_hub.compat` without importing any reader machinery. A
report describes structure only: node kinds and parameter shapes, or key names
and tensor shapes. No tensor value ever survives into it.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

#: A persisted NIR graph this project or another framework produced.
NIR_GRAPH = "nir_graph"
#: A torch checkpoint or bare state dict.
STATE_DICT = "state_dict"
#: Framework weights in a declared but structurally opaque format.
FRAMEWORK_WEIGHTS = "framework_weights"

#: File suffixes recognised as each artifact kind.
NIR_SUFFIXES = (".json",)
TORCH_SUFFIXES = (".pt", ".pth", ".ckpt")
WEIGHT_SUFFIXES = (".bin", ".safetensors", ".npz", ".h5", ".hdf5")


@dataclass(frozen=True)
class ArtifactReport:
    """A structural description of one artifact, before any load."""

    kind: str
    path: str
    nodes: Tuple[Dict[str, Any], ...]
    keys: Tuple[Dict[str, Any], ...]
    edges: Tuple[Dict[str, Any], ...]
    notes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-able inspect payload the socket and CLI emit."""
        return {
            "kind": self.kind,
            "path": self.path,
            "nodes": [dict(node) for node in self.nodes],
            "keys": [dict(key) for key in self.keys],
            "edges": [dict(edge) for edge in self.edges],
            "notes": list(self.notes),
        }


def _dims(value: Sequence[Any]) -> list:
    """Return the nested dimensions of a list-like parameter value."""
    dims = [len(value)]
    first = value[0] if value else None
    if isinstance(first, (list, tuple)):
        dims.extend(_dims(first))
    return dims


def _shape(value: Any) -> Any:
    """Return a scalar, or the nested dimensions of an array value."""
    if isinstance(value, (list, tuple)):
        return _dims(value)
    return value


def normalize_nodes(
    nodes: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    """Return node records with array parameters reduced to their shapes."""
    return tuple(
        {
            "name": str(node["name"]),
            "kind": str(node["kind"]),
            "params": {
                str(key): _shape(value)
                for key, value in dict(node.get("params", {})).items()
            },
        }
        for node in nodes
    )
