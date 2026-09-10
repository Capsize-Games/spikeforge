"""Structural validation for topology graphs."""

from typing import List, Sequence, Set

from snn_interpreter.topology.edge import Edge
from snn_interpreter.topology.ordering import (
    forward_edges,
    topological_order,
)
from snn_interpreter.topology.stage import Stage


def assert_stage_exists(name: str, stages: Sequence[Stage]) -> None:
    """Raise when ``name`` is not among ``stages``."""
    if name not in {stage.name for stage in stages}:
        raise ValueError(f"unknown stage: {name!r}")


def assert_references(
    stages: Sequence[Stage], edges: Sequence[Edge]
) -> None:
    """Raise when an edge names a stage that does not exist."""
    names = {stage.name for stage in stages}
    for edge in edges:
        if edge.source not in names or edge.target not in names:
            raise ValueError(
                f"edge {edge.source!r} -> {edge.target!r} references an "
                "unknown stage"
            )


def assert_acyclic(
    stages: Sequence[Stage], edges: Sequence[Edge]
) -> None:
    """Raise when the forward (non-delayed) graph contains a cycle."""
    topological_order(stages, edges)


def _reachable(
    source: str, forward: Sequence[Edge]
) -> Set[str]:
    seen: Set[str] = {source}
    frontier: List[str] = [source]
    while frontier:
        current = frontier.pop()
        for edge in forward:
            if edge.source == current and edge.target not in seen:
                seen.add(edge.target)
                frontier.append(edge.target)
    return seen


def assert_reachable(
    source: str, stages: Sequence[Stage], edges: Sequence[Edge]
) -> None:
    """Raise when a stage has no forward path from ``source``."""
    seen = _reachable(source, forward_edges(edges))
    missing = [stage.name for stage in stages if stage.name not in seen]
    if missing:
        raise ValueError(
            f"stages unreachable from {source!r}: {missing}"
        )
