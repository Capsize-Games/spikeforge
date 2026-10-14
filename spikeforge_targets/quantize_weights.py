"""The weight-quantization pass over a NIR graph.

Split out of :mod:`spikeforge_targets.quantize` so that module stays within
the file-length limit. A weight scheme restricts every weight-bearing node
in a fresh graph, recording per-layer before/after ranges; an unknown scheme
is reported unapplied rather than guessed.
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from spikeforge.nir_bridge.require import require_node
from spikeforge_targets.quantize_report import Layers
from spikeforge_targets.quantize_schemes import SCHEMES, Scheme, node_weight

#: Reason recorded when a target declares no weight quantization at all.
NO_QUANTIZATION = "target declares no quantization"
#: Reason recorded for a weight scheme name this module cannot apply.
UNKNOWN_SCHEME = "unknown quantization scheme {scheme!r}"

_Layer = Dict[str, Any]
#: The weight pass: the graph to run, its layer records, applied, reason.
_Weights = Tuple[Any, Layers, bool, str]


def _layer(name: str, node: Any, before: Any, after: Any) -> _Layer:
    """Return one per-layer before/after range record."""
    return {
        "node": name,
        "primitive": type(node).__name__,
        "before": [float(before[0]), float(before[1])],
        "after": [float(after[0]), float(after[1])],
    }


def _one(name: str, node: Any, scheme: Scheme) -> Tuple[Any, Optional[_Layer]]:
    """Return ``node`` with quantized weights and its layer record."""
    weight = node_weight(node)
    if weight is None:
        return node, None
    quantized, before, after = scheme(weight)
    return replace(node, weight=quantized), _layer(name, node, before, after)


def _nodes(graph: Any, scheme: Scheme) -> Tuple[Dict[str, Any], List[_Layer]]:
    """Return quantized nodes and their per-layer records."""
    nodes: Dict[str, Any] = {}
    layers: List[_Layer] = []
    for name, node in graph.nodes.items():
        new_node, record = _one(name, node, scheme)
        nodes[name] = new_node
        if record is not None:
            layers.append(record)
    return nodes, layers


def _new_graph(nodes: Dict[str, Any], edges: Any) -> Any:
    """Return a fresh, unchecked NIR graph from ``nodes`` and ``edges``."""
    cls = require_node("NIRGraph", "graph")
    return cls(dict(nodes), list(edges), type_check=False)


def _weights(graph: Any, scheme_name: str) -> _Weights:
    """Return the weight-quantized graph, or ``graph`` itself with a reason."""
    if scheme_name == "none":
        return graph, (), False, NO_QUANTIZATION
    scheme = SCHEMES.get(scheme_name)
    if scheme is None:
        return graph, (), False, UNKNOWN_SCHEME.format(scheme=scheme_name)
    nodes, layers = _nodes(graph, scheme)
    return _new_graph(nodes, graph.edges), tuple(layers), True, ""
