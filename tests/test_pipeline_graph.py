"""The pipeline DAG data model: parsing, validation, topological order."""

from typing import Sequence

import pytest

from spikeforge_serve.pipeline import (
    PipelineEdge,
    PipelineGraph,
    PipelineGraphError,
    PipelineNode,
)


def _graph(
    nodes: Sequence[PipelineNode],
    edges: Sequence[PipelineEdge],
    name: str = "g",
) -> PipelineGraph:
    """Build a PipelineGraph from plain sequences, for terser test bodies."""
    return PipelineGraph(name=name, nodes=tuple(nodes), edges=tuple(edges))


def test_round_trips_through_dict() -> None:
    """to_dict/from_dict is lossless for a well-formed graph."""
    graph = _graph(
        [
            PipelineNode("n1", "digits", {"x": 1.0, "y": 2.0}),
            PipelineNode("n2", "risk"),
        ],
        [PipelineEdge("e1", "n1", "n2", "predicted_class")],
    )
    restored = PipelineGraph.from_dict(graph.to_dict())
    assert restored == graph


def test_linear_pipeline_orders_source_before_target() -> None:
    """A -> B always places A before B."""
    graph = _graph(
        [PipelineNode("b", "ckpt_b"), PipelineNode("a", "ckpt_a")],
        [PipelineEdge("e1", "a", "b")],
    )
    order = [node.id for node in graph.topological_order()]
    assert order == ["a", "b"]


def test_disconnected_nodes_both_appear() -> None:
    """Two source nodes with no edge between them both come out."""
    graph = _graph(
        [PipelineNode("a", "ckpt_a"), PipelineNode("b", "ckpt_b")], []
    )
    order = {node.id for node in graph.topological_order()}
    assert order == {"a", "b"}


def test_cycle_is_rejected() -> None:
    """A -> B -> A raises instead of looping or silently truncating."""
    graph = _graph(
        [PipelineNode("a", "ckpt_a"), PipelineNode("b", "ckpt_b")],
        [PipelineEdge("e1", "a", "b"), PipelineEdge("e2", "b", "a")],
    )
    with pytest.raises(PipelineGraphError, match="cycle"):
        graph.topological_order()


def test_edge_to_unknown_node_is_rejected() -> None:
    """An edge naming a node id that doesn't exist is a clear error."""
    graph = _graph(
        [PipelineNode("a", "ckpt_a")],
        [PipelineEdge("e1", "a", "ghost")],
    )
    with pytest.raises(PipelineGraphError, match="unknown target"):
        graph.validate()


def test_duplicate_node_id_is_rejected() -> None:
    """Two nodes sharing an id is malformed, not silently deduplicated."""
    graph = _graph(
        [PipelineNode("a", "ckpt_a"), PipelineNode("a", "ckpt_b")], []
    )
    with pytest.raises(PipelineGraphError, match="duplicate"):
        graph.validate()


def test_unknown_extract_mode_is_rejected_at_construction() -> None:
    """An edge outside the fixed extract enum fails immediately."""
    with pytest.raises(PipelineGraphError, match="extract mode"):
        PipelineEdge("e1", "a", "b", extract="whatever_i_want")


def test_incoming_returns_only_edges_targeting_the_node() -> None:
    """incoming() filters by target, not by source."""
    graph = _graph(
        [
            PipelineNode("a", "x"),
            PipelineNode("b", "y"),
            PipelineNode("c", "z"),
        ],
        [PipelineEdge("e1", "a", "b"), PipelineEdge("e2", "a", "c")],
    )
    assert [e.id for e in graph.incoming("b")] == ["e1"]
    assert [e.id for e in graph.incoming("a")] == []
