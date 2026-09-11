"""Shared helpers for lowering a NIR graph onto a backend.

The Norse and Lava backends both execute a single ordered chain of layers,
so the walk that recovers that chain from a graph, and the scalar extraction
used to read node parameters, live here once. A graph that branches, merges,
or is disconnected is rejected with a named reason so the backend reports an
honest error instead of guessing at an execution order.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

#: A ``(node name, nir node)`` pair in execution order.
Layer = Tuple[str, Any]


def scalar(value: Any) -> float:
    """Return a node parameter as a Python float."""
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _adjacency(
    edges: List[Tuple[str, str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return the successor and predecessor maps of ``edges``."""
    successors: Dict[str, List[str]] = {}
    predecessors: Dict[str, List[str]] = {}
    for source, target in edges:
        successors.setdefault(source, []).append(target)
        predecessors.setdefault(target, []).append(source)
    return successors, predecessors


def _walk(
    start: str, successors: Dict[str, List[str]], nodes: Dict[str, Any]
) -> List[Layer]:
    """Walk the single successor chain from ``start`` to its sink."""
    chain: List[Layer] = []
    current: Any = start
    while current is not None:
        chain.append((current, nodes[current]))
        outs = successors.get(current, [])
        if len(outs) > 1:
            raise ValueError(
                f"backend lowering does not support branch at {current!r}"
            )
        current = outs[0] if outs else None
    return chain


def linear_chain(graph: Any) -> List[Layer]:
    """Return the graph's layers in order, or raise a named reason.

    Only a single source-to-sink chain is executable by the lowered
    backends; a merge, a branch, or a cycle is rejected here.
    """
    names = list(graph.nodes)
    successors, predecessors = _adjacency(list(graph.edges))
    sources = [name for name in names if not predecessors.get(name)]
    if len(sources) != 1:
        raise ValueError("backend lowering needs exactly one source node")
    chain = _walk(sources[0], successors, dict(graph.nodes))
    if len(chain) != len(names):
        raise ValueError("backend lowering needs a single linear chain")
    return chain
