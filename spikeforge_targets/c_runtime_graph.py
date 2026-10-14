"""Graph-level conformance of an export summary against the C loader.

Mirrors the structural assumptions in ``nir-c-runtime`` ``src/graph.c``
(edges name existing nodes) and ``src/runtime.c`` (every evaluated node
sums at least one incoming edge; the reduced graph, ignoring edges into
``Delay`` nodes, is acyclic). Export always emits exactly one ``Input``
and one ``Output`` node, which the checks pin down.
"""

from typing import Dict, List, Sequence, Set, Tuple

from spikeforge_targets.c_runtime_violation import CRuntimeViolation

#: A ``(node name, node kind)`` pair in summary order.
NodeKind = Tuple[str, str]
#: A ``(source name, target name)`` edge.
Edge = Tuple[str, str]


def _duplicates(nodes: Sequence[NodeKind]) -> List[CRuntimeViolation]:
    """Return one violation per repeated node name."""
    seen: Set[str] = set()
    found: List[CRuntimeViolation] = []
    for name, kind in nodes:
        if name in seen:
            found.append(
                CRuntimeViolation(
                    name=name,
                    kind=kind,
                    reason=(
                        "duplicate node name; the C loader resolves "
                        "edges to the first match"
                    ),
                )
            )
        seen.add(name)
    return found


def _unknown_refs(
    known: Set[str], edges: Sequence[Edge]
) -> List[CRuntimeViolation]:
    """Return one violation per edge naming an absent node."""
    found: List[CRuntimeViolation] = []
    for source, target in edges:
        for missing in (source, target):
            if missing not in known:
                found.append(
                    CRuntimeViolation(
                        name=f"{source}->{target}",
                        kind="edge",
                        reason=(
                            f"edge references an unknown node {missing!r}"
                        ),
                    )
                )
    return found


def _missing_inputs(
    nodes: Sequence[NodeKind], edges: Sequence[Edge]
) -> List[CRuntimeViolation]:
    """Return one violation per starved non-Input node."""
    fed = {target for _, target in edges}
    return [
        CRuntimeViolation(
            name=name,
            kind=kind,
            reason=(
                "node has no incoming edge; the C runtime sums "
                "incoming edges and fails on none"
            ),
        )
        for name, kind in nodes
        if kind != "Input" and name not in fed
    ]


def _endpoint_counts(nodes: Sequence[NodeKind]) -> List[CRuntimeViolation]:
    """Return violations unless Input/Output appear exactly once each."""
    found: List[CRuntimeViolation] = []
    for kind in ("Input", "Output"):
        count = sum(1 for _, item in nodes if item == kind)
        if count != 1:
            found.append(
                CRuntimeViolation(
                    name="graph",
                    kind="graph",
                    reason=(
                        f"graph has {count} {kind} nodes; the C "
                        "runtime runs a single input to a readout"
                    ),
                )
            )
    return found


def _reduced(edges: Sequence[Edge], delays: Set[str]) -> List[Edge]:
    """Return edges ignoring those feeding a Delay node."""
    return [
        (source, target) for source, target in edges if target not in delays
    ]


def _known_edges(
    nodes: Sequence[NodeKind], edges: Sequence[Edge]
) -> List[Edge]:
    """Return the edges whose endpoints are both known nodes."""
    known = {name for name, _ in nodes}
    return [
        (source, target)
        for source, target in edges
        if source in known and target in known
    ]


def _has_cycle(nodes: Sequence[NodeKind], edges: Sequence[Edge]) -> bool:
    """Return True when the reduced graph still holds a cycle."""
    delays = {name for name, kind in nodes if kind == "Delay"}
    reduced = _reduced(_known_edges(nodes, edges), delays)
    pending: Dict[str, int] = {name: 0 for name, _ in nodes}
    for _, target in reduced:
        pending[target] += 1
    ready = [name for name, count in pending.items() if count == 0]
    drained = 0
    children: Dict[str, List[str]] = {name: [] for name, _ in nodes}
    for source, target in reduced:
        children[source].append(target)
    while ready:
        drained += 1
        for child in children[ready.pop(0)]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    return drained != len(pending)


def check_graph(
    nodes: Sequence[NodeKind], edges: Sequence[Edge]
) -> List[CRuntimeViolation]:
    """Return every graph-level violation, in deterministic order."""
    known = {name for name, _ in nodes}
    found = _duplicates(nodes)
    found.extend(_unknown_refs(known, edges))
    found.extend(_missing_inputs(nodes, edges))
    found.extend(_endpoint_counts(nodes))
    if _has_cycle(nodes, edges):
        found.append(
            CRuntimeViolation(
                name="graph",
                kind="graph",
                reason="NIR graph cycle is not broken by a Delay node",
            )
        )
    return found
