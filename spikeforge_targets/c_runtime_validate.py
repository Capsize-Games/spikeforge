"""Validate an export against the ``nir-c-runtime`` profile.

:func:`validate_for_c_runtime` accepts a :class:`TopologySpec`, an
exported NIR graph, or an existing :func:`graph_summary` mapping and
returns a :class:`CRuntimeReport` naming every node, edge, and
structural assumption the C loader would reject.
:func:`check_for_c_runtime` is the refuse-early gate: it raises the
typed :class:`CRuntimeUnsupportedError` on the same violations.

``spikeforge`` imports stay inside the functions so this module never
pulls ``torch`` at import time; summary mappings validate with the
standard library alone.
"""

from typing import Any, Dict, List, Mapping, Tuple

from spikeforge_targets.c_runtime import TARGET_NAME
from spikeforge_targets.c_runtime_errors import CRuntimeUnsupportedError
from spikeforge_targets.c_runtime_graph import Edge, NodeKind, check_graph
from spikeforge_targets.c_runtime_report import CRuntimeReport
from spikeforge_targets.c_runtime_shapes import check_node
from spikeforge_targets.c_runtime_violation import CRuntimeViolation


def _node_pairs(
    summary: Mapping[str, Any],
) -> Tuple[List[NodeKind], Dict[str, Mapping[str, Any]]]:
    """Return the node pairs and params of a summary mapping."""
    nodes: List[NodeKind] = []
    params: Dict[str, Mapping[str, Any]] = {}
    for item in summary["nodes"]:
        nodes.append(_one_node(item, params))
    return nodes, params


def _one_node(item: Any, params: Dict[str, Mapping[str, Any]]) -> NodeKind:
    """Record one summary node, returning its ``(name, kind)`` pair."""
    if not isinstance(item, Mapping):
        raise ValueError("graph summary node records must be mappings")
    name = item.get("name")
    kind = item.get("kind")
    if not isinstance(name, str) or not isinstance(kind, str):
        raise ValueError("graph summary nodes need a name and a kind")
    node_params = item.get("params", {})
    params[name] = node_params if isinstance(node_params, Mapping) else {}
    return (name, kind)


def _edge_pairs(summary: Mapping[str, Any]) -> List[Edge]:
    """Return the ``(source, target)`` edges of a summary mapping."""
    edges: List[Edge] = []
    for item in summary["edges"]:
        if not isinstance(item, Mapping):
            raise ValueError("graph summary edge records must be mappings")
        source = item.get("source")
        target = item.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("graph summary edges need source and target")
        edges.append((source, target))
    return edges


def _as_summary(target: Any) -> Mapping[str, Any]:
    """Return ``target`` as a summary mapping, exporting when needed."""
    if isinstance(target, Mapping):
        nodes = target.get("nodes")
        edges = target.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError(
                "expected a graph_summary mapping with nodes and edges"
            )
        return target
    from spikeforge.nir_bridge.exporter import graph_summary

    if hasattr(target, "nodes") and hasattr(target, "edges"):
        return graph_summary(target)
    from spikeforge.topology.spec import TopologySpec

    if isinstance(target, TopologySpec):
        from spikeforge.nir_bridge.exporter import to_nir

        return graph_summary(to_nir(target))
    raise ValueError(
        "expected a TopologySpec, a NIR graph, or a summary mapping"
    )


def _node_violations(
    nodes: List[NodeKind], params: Dict[str, Mapping[str, Any]]
) -> Tuple[List[str], List[CRuntimeViolation]]:
    """Return the supported names and node violations of ``nodes``."""
    supported: List[str] = []
    found: List[CRuntimeViolation] = []
    for name, kind in nodes:
        violation = check_node(name, kind, params.get(name, {}))
        if violation is None:
            supported.append(name)
        else:
            found.append(violation)
    return supported, found


def validate_for_c_runtime(target: Any) -> CRuntimeReport:
    """Return the C-runtime conformance report for ``target``.

    ``target`` may be a :class:`TopologySpec` (exported structurally),
    an exported NIR graph, or an existing summary mapping. Node checks
    run before graph checks so an unsupported kind is reported alongside
    any edge or structural gap, each exactly once.
    """
    summary = _as_summary(target)
    nodes, params = _node_pairs(summary)
    edges = _edge_pairs(summary)
    supported, found = _node_violations(nodes, params)
    found.extend(check_graph(nodes, edges))
    return CRuntimeReport(
        target=TARGET_NAME,
        supported=tuple(supported),
        unsupported=tuple(found),
    )


def check_for_c_runtime(target: Any) -> None:
    """Raise :class:`CRuntimeUnsupportedError` unless ``target`` conforms."""
    report = validate_for_c_runtime(target)
    if not report.passed():
        raise CRuntimeUnsupportedError(report.unsupported)
