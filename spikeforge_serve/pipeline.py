"""The pipeline graph data model: nodes are checkpoints, edges carry data.

A pipeline is a DAG, not a state machine: nodes run once each, in
topological order, and a node's input is entirely determined by its
incoming edge's `extract` of the upstream node's output. No cycles, no
conditional transitions -- see documentation/model-deployment.md for why
that's a deliberate v1 scope cut rather than an oversight.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

#: Fixed set of ways an edge can shape its source node's output into the
#: target node's input. Not a general expression language by design: it
#: covers real chaining (the mean-readout vector is already frame-shaped)
#: without needing a JSONPath parser.
EXTRACT_MODES: Tuple[str, ...] = ("mean_logits", "predicted_class", "one_hot")

_DEFAULT_POSITION = {"x": 0.0, "y": 0.0}


class PipelineGraphError(ValueError):
    """A pipeline graph is malformed: an unknown node, a cycle, fan-in."""


@dataclass(frozen=True)
class PipelineNode:
    """One pipeline node: a saved checkpoint plus its canvas position."""

    id: str
    checkpoint: str
    position: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_POSITION)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-ready form of this node."""
        return {
            "id": self.id,
            "checkpoint": self.checkpoint,
            "position": dict(self.position),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineNode":
        """Parse one node from its JSON form."""
        return cls(
            id=str(data["id"]),
            checkpoint=str(data["checkpoint"]),
            position=dict(data.get("position") or _DEFAULT_POSITION),
        )


@dataclass(frozen=True)
class PipelineEdge:
    """One edge: the source node's output feeds the target via `extract`."""

    id: str
    source: str
    target: str
    extract: str = "mean_logits"

    def __post_init__(self) -> None:
        """Reject an edge naming an extract mode outside the fixed set."""
        if self.extract not in EXTRACT_MODES:
            raise PipelineGraphError(
                f"edge {self.id!r}: unknown extract mode {self.extract!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-ready form of this edge."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "extract": self.extract,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineEdge":
        """Parse one edge from its JSON form."""
        return cls(
            id=str(data["id"]),
            source=str(data["source"]),
            target=str(data["target"]),
            extract=str(data.get("extract", "mean_logits")),
        )


@dataclass(frozen=True)
class PipelineGraph:
    """A DAG of checkpoints: nodes plus the edges wiring their outputs."""

    name: str
    nodes: Tuple[PipelineNode, ...]
    edges: Tuple[PipelineEdge, ...]
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-ready form of the whole graph."""
        return {
            "version": self.version,
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineGraph":
        """Parse and validate a graph from its JSON form."""
        nodes = tuple(
            PipelineNode.from_dict(item) for item in data.get("nodes", [])
        )
        edges = tuple(
            PipelineEdge.from_dict(item) for item in data.get("edges", [])
        )
        graph = cls(
            name=str(data.get("name", "")),
            nodes=nodes,
            edges=edges,
            version=int(data.get("version", 1)),
        )
        graph.validate()
        return graph

    def validate(self) -> None:
        """Raise PipelineGraphError on a duplicate or unknown node id."""
        ids = [node.id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise PipelineGraphError("pipeline graph has a duplicate node id")
        id_set = set(ids)
        for edge in self.edges:
            if edge.source not in id_set:
                raise PipelineGraphError(
                    f"edge {edge.id!r}: unknown source {edge.source!r}"
                )
            if edge.target not in id_set:
                raise PipelineGraphError(
                    f"edge {edge.id!r}: unknown target {edge.target!r}"
                )

    def incoming(self, node_id: str) -> List[PipelineEdge]:
        """Return every edge whose target is ``node_id``."""
        return [edge for edge in self.edges if edge.target == node_id]

    def topological_order(self) -> List[PipelineNode]:
        """Return nodes ordered so every source precedes its targets.

        Kahn's algorithm: repeatedly take a zero-in-degree node, then
        decrement its successors' in-degree. If nodes remain once no
        zero-in-degree node is left, they form a cycle.
        """
        self.validate()
        ordered = _kahn_order(self.nodes, self.edges)
        if len(ordered) != len(self.nodes):
            raise PipelineGraphError("pipeline graph contains a cycle")
        return ordered


def _in_degrees(
    nodes: Tuple[PipelineNode, ...], edges: Tuple[PipelineEdge, ...]
) -> Dict[str, int]:
    """Return each node id's in-degree (count of incoming edges)."""
    in_degree = {node.id: 0 for node in nodes}
    for edge in edges:
        in_degree[edge.target] += 1
    return in_degree


def _kahn_order(
    nodes: Tuple[PipelineNode, ...], edges: Tuple[PipelineEdge, ...]
) -> List[PipelineNode]:
    """Return ``nodes`` in a zero-in-degree-first topological order."""
    by_id = {node.id: node for node in nodes}
    in_degree = _in_degrees(nodes, edges)
    ready = sorted(
        node_id for node_id, degree in in_degree.items() if degree == 0
    )
    remaining_edges = list(edges)
    ordered: List[PipelineNode] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(by_id[node_id])
        _release_targets(node_id, remaining_edges, in_degree, ready)
        ready.sort()
    return ordered


def _release_targets(
    node_id: str,
    remaining_edges: List[PipelineEdge],
    in_degree: Dict[str, int],
    ready: List[str],
) -> None:
    """Drop ``node_id``'s outgoing edges, queuing newly zero-in-degree ids."""
    outgoing = [e for e in remaining_edges if e.source == node_id]
    for edge in outgoing:
        remaining_edges.remove(edge)
        in_degree[edge.target] -= 1
        if in_degree[edge.target] == 0:
            ready.append(edge.target)
