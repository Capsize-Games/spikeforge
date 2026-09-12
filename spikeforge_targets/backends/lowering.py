"""Shared helpers for lowering a NIR graph onto a backend.

The Norse, Lava, and vendor simulator backends all execute a single ordered
chain of layers, so the walk that recovers that chain from a graph, the scalar
extraction used to read node parameters, and the dense/neuron program lowering
live here once. A graph that branches, merges, or is disconnected is rejected
with a named reason so the backend reports an honest error instead of guessing
at an execution order.
"""

from math import exp
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: A ``(node name, nir node)`` pair in execution order.
Layer = Tuple[str, Any]
#: Node kinds a lowered program expresses as a dense linear layer.
LINEAR_KINDS = ("Affine", "Linear")
#: Node kinds a lowered program expresses as a neuron layer.
NEURON_KINDS = ("LIF", "LI")


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


def _linear_layer(node: Any) -> Dict[str, Any]:
    """Return a dense layer description from a linear node."""
    weight = np.asarray(node.weight, dtype=np.float32)
    return {
        "kind": "Linear",
        "params": {"weight": weight},
        "size": int(weight.shape[0]),
    }


def _neuron_layer(
    kind: str, node: Any, width: Optional[int]
) -> Dict[str, Any]:
    """Return a LIF/LI layer description at the tracked chain width."""
    if width is None:
        raise ValueError(f"lowering cannot size a leading {kind!r} node")
    decay = exp(-1.0 / scalar(node.tau))
    return {
        "kind": kind,
        "params": {
            "decay": 1.0 - decay,
            "v_threshold": scalar(node.v_threshold),
        },
        "size": width,
    }


def linear_program(graph: Any) -> Dict[str, Any]:
    """Lower a linear graph to a dense/neuron layer program, or raise.

    Only a ``<dense>`` / ``<neuron>`` chain is expressible: a leading neuron
    has no tracked width, and a convolution, pooling, or other node is refused
    with the offending node and kind named rather than silently approximated.
    """
    layers: List[Dict[str, Any]] = []
    width: Optional[int] = None
    for name, node in linear_chain(graph):
        kind = type(node).__name__
        if kind in ("Input", "Output"):
            continue
        if kind in LINEAR_KINDS:
            layer = _linear_layer(node)
            width = layer["size"]
        elif kind in NEURON_KINDS:
            layer = _neuron_layer(kind, node, width)
        else:
            raise ValueError(
                f"lowering does not support node {name!r} of kind {kind!r}"
            )
        layers.append(layer)
    return {"layers": layers}
