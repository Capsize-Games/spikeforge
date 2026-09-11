"""Export a topology spec as an NIR graph and a JSON-able summary.

``to_nir`` renders a :class:`TopologySpec` into a ``nir.NIRGraph``.
Weighted stages copy their learned tensors from the supplied built module;
without a module they fall back to zero-initialised placeholder weights so
a purely structural graph is still coherent.

Type checking is disabled on the emitted graph (matching
``nirtorch.extract_nir_graph``) because pooling and convolution nodes can
only be type-inferred once a concrete input shape is known.

``graph_summary`` returns a tensor-free node/edge description suitable for
streaming over JSON; it accepts either a spec or an already-built graph.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from snn_interpreter.nir_bridge.jsonable import node_params
from snn_interpreter.nir_bridge.planner import build_plan
from snn_interpreter.nir_bridge.require import require_node
from snn_interpreter.topology.spec import TopologySpec


def to_nir(spec: TopologySpec, module: Optional[Any] = None) -> Any:
    """Return the NIR graph rendering ``spec``.

    ``module`` is the built ``StageModule`` whose learned weights are copied
    into the graph. When omitted, weighted stages carry zero placeholders
    and the result is a structural graph only.
    """
    nodes, edges = build_plan(spec, module)
    graph_cls = require_node("NIRGraph", "graph")
    node_map = {mapping.name: mapping.node for mapping in nodes}
    return graph_cls(node_map, list(edges), type_check=False)


def _is_delay(node: Optional[Any]) -> bool:
    """Return True when ``node`` is a NIR ``Delay`` node."""
    return node is not None and type(node).__name__ == "Delay"


def _summary_source(
    target: Any,
) -> Tuple[List[str], List[Any], List[Tuple[str, str]]]:
    """Return ``(names, nodes, edges)`` for a spec or a NIR graph."""
    if isinstance(target, TopologySpec):
        plan_nodes, plan_edges = build_plan(target)
        names = [mapping.name for mapping in plan_nodes]
        return (
            names,
            [mapping.node for mapping in plan_nodes],
            list(plan_edges),
        )
    names = list(target.nodes)
    return names, [target.nodes[name] for name in names], list(target.edges)


def _node_summary(
    names: Sequence[str], nodes: Sequence[Any]
) -> List[Dict[str, Any]]:
    """Return one ``{name, kind, params}`` record per node."""
    return [
        {
            "name": name,
            "kind": type(node).__name__,
            "params": node_params(node),
        }
        for name, node in zip(names, nodes)
    ]


def _edge_summary(
    edges: Sequence[Tuple[str, str]], lookup: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Return one ``{source, target, delayed}`` record per edge."""
    return [
        {
            "source": source,
            "target": sink,
            "delayed": _is_delay(lookup.get(sink)),
        }
        for source, sink in edges
    ]


def graph_summary(target: Any) -> Dict[str, Any]:
    """Return a JSON-serialisable node/edge description.

    ``target`` may be a :class:`TopologySpec` or an exported NIR graph. Node
    parameters are plain numbers, strings and lists only; no tensor object
    survives into the result. An edge is flagged ``delayed`` when it feeds a
    NIR ``Delay`` node.
    """
    names, nodes, edges = _summary_source(target)
    lookup = dict(zip(names, nodes))
    return {
        "nodes": _node_summary(names, nodes),
        "edges": _edge_summary(edges, lookup),
    }
