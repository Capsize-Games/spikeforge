"""Topological ordering of the forward (non-delayed) topology edges."""

from typing import Dict, List, Sequence

from spikeforge.topology.edge import Edge
from spikeforge.topology.stage import Stage


def forward_edges(edges: Sequence[Edge]) -> List[Edge]:
    """Return only the non-delayed edges that define forward flow."""
    return [edge for edge in edges if not edge.delayed]


def _inbound_counts(
    names: Sequence[str], forward: Sequence[Edge]
) -> Dict[str, int]:
    counts: Dict[str, int] = dict.fromkeys(names, 0)
    for edge in forward:
        counts[edge.target] += 1
    return counts


def _children(name: str, forward: Sequence[Edge]) -> List[str]:
    return [edge.target for edge in forward if edge.source == name]


def topological_order(
    stages: Sequence[Stage], edges: Sequence[Edge]
) -> List[str]:
    """Return stage names in dependency order; raise on a forward cycle."""
    names = [stage.name for stage in stages]
    forward = forward_edges(edges)
    inbound = _inbound_counts(names, forward)
    ready = [name for name in names if inbound[name] == 0]
    order: List[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for child in _children(name, forward):
            inbound[child] -= 1
            if inbound[child] == 0:
                ready.append(child)
    if len(order) != len(names):
        raise ValueError("topology contains a cycle among forward edges")
    return order
