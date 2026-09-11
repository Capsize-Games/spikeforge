"""Extract ``(name, kind)`` pairs from a graph, spec, or summary."""

from typing import Any, List, Mapping, Tuple

from spikeforge.nir_bridge.exporter import graph_summary, to_nir
from spikeforge.topology.spec import TopologySpec

#: A ``(node name, primitive class name)`` pair.
NodeKind = Tuple[str, str]


def _summary_pairs(summary: Mapping[str, Any]) -> List[NodeKind]:
    """Return the node pairs of an already-summarised graph."""
    return [
        (str(item["name"]), str(item["kind"])) for item in summary["nodes"]
    ]


def node_kinds(target: Any) -> List[NodeKind]:
    """Return ``(name, kind)`` for every node in ``target``.

    ``target`` may be a :class:`TopologySpec` (rendered with ``to_nir``), an
    exported NIR graph, or an existing :func:`graph_summary` mapping. The
    list order follows the graph's own node order.
    """
    if isinstance(target, TopologySpec):
        return _summary_pairs(graph_summary(to_nir(target)))
    if isinstance(target, Mapping):
        return _summary_pairs(target)
    names = list(target.nodes)
    return [
        (str(name), type(target.nodes[name]).__name__) for name in names
    ]
