"""Execute a target's declared substitutions to build a target-ready graph.

The declared :attr:`TargetSpec.substitutions` mapping stays the source of
truth for *which* rewrites apply; this module only consumes it. Every node
the target does not support natively lands in exactly one report bucket: a
declared and executable substitution is **applied**, a declared substitution
with no rule is **skipped**, and a primitive with no declaration is
**unfixable**. Nothing is dropped, and the original graph is left untouched.

When a spike fixture is supplied the rewritten graph is executed by the
reference interpreter alongside the original and the measured drift is folded
into the report, so a lossy substitution is quantified rather than hidden.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from spikeforge.nir_bridge import api
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology.spec import TopologySpec
from spikeforge_targets.registry import get_target
from spikeforge_targets.rewrite_drift import rewrite_drift
from spikeforge_targets.rewrite_report import RewriteReport
from spikeforge_targets.rewrite_result import RewriteResult
from spikeforge_targets.substitute_ops import apply_rewrite
from spikeforge_targets.target_spec import TargetSpec

#: Reason recorded for a node whose primitive has no declared substitution.
NO_SUBSTITUTION = "no substitution declared"
#: Reason recorded for a declared substitution that has no executable rule.
NO_RULE = "no executable rule for {primitive}->{substitute}"

_Remap = Tuple[str, str]


def _resolve(target: Any) -> TargetSpec:
    """Return the :class:`TargetSpec` for a name or a spec object."""
    if isinstance(target, TargetSpec):
        return target
    return get_target(str(target))


def _as_graph(graph_or_spec: Any) -> Any:
    """Return a NIR graph, exporting a :class:`TopologySpec` when given one."""
    if isinstance(graph_or_spec, TopologySpec):
        return to_nir(graph_or_spec)
    return graph_or_spec


def _record(name: str, kind: str, form: str, detail: str) -> Dict[str, str]:
    """Return an applied-rewrite record for ``name``."""
    return {"node": name, "from": kind, "to": form, "detail": detail}


def _unfixable(name: str, kind: str) -> Dict[str, str]:
    """Return the unfixable record for a primitive with no substitution."""
    return {"node": name, "primitive": kind, "reason": NO_SUBSTITUTION}


def _skipped(name: str, kind: str, form: str) -> Dict[str, str]:
    """Return the skipped record for a declared but unexecutable rule."""
    reason = NO_RULE.format(primitive=kind, substitute=form)
    return {"node": name, "from": kind, "to": form, "reason": reason}


def _rewrite_one(
    name: str, node: Any, spec: TargetSpec, acc: Dict[str, Any]
) -> None:
    """Place one node's rendering into the accumulator, or record why."""
    kind = type(node).__name__
    if spec.supports(kind):
        acc["nodes"][name] = node
        return
    form = spec.substitute(kind)
    if form is None:
        acc["unfixable"].append(_unfixable(name, kind))
        acc["nodes"][name] = node
        return
    outcome = apply_rewrite(kind, form, name, node)
    if outcome is None:
        acc["skipped"].append(_skipped(name, kind, form))
        acc["nodes"][name] = node
        return
    for new_name, new_node in outcome.nodes:
        acc["nodes"][new_name] = new_node
    acc["internal"].extend(outcome.edges)
    acc["remap"][name] = (outcome.entry, outcome.output)
    acc["applied"].append(_record(name, kind, form, outcome.detail))


def _rewire(
    edges: Sequence[Tuple[str, str]],
    remap: Mapping[str, _Remap],
    internal: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Return ``edges`` with rewritten nodes replaced by their new handles."""
    rewired: List[Tuple[str, str]] = []
    for source, target in edges:
        start = remap[source][1] if source in remap else source
        end = remap[target][0] if target in remap else target
        rewired.append((start, end))
    return list(dict.fromkeys([*rewired, *internal]))


def _new_graph(nodes: Mapping[str, Any], edges: Sequence[Any]) -> Any:
    """Return a fresh, unchecked NIR graph from ``nodes`` and ``edges``."""
    cls = api.node_class("NIRGraph")
    return cls(dict(nodes), list(edges), type_check=False)


def _accumulator() -> Dict[str, Any]:
    """Return an empty rewrite accumulator."""
    return {
        "nodes": {},
        "internal": [],
        "remap": {},
        "applied": [],
        "skipped": [],
        "unfixable": [],
    }


def rewrite(
    graph_or_spec: Any, target: Any, spikes: Optional[Any] = None
) -> RewriteResult:
    """Return the target-ready graph for ``target`` plus its rewrite report.

    ``graph_or_spec`` is a NIR graph or a :class:`TopologySpec`; ``target`` a
    name or :class:`TargetSpec`. Supplying ``spikes`` adds the post-rewrite
    drift check, executing both graphs with the reference interpreter.
    """
    spec = _resolve(target)
    graph = _as_graph(graph_or_spec)
    acc = _accumulator()
    for name, node in graph.nodes.items():
        _rewrite_one(name, node, spec, acc)
    edges = _rewire(graph.edges, acc["remap"], acc["internal"])
    ready = _new_graph(acc["nodes"], edges)
    drift = None if spikes is None else rewrite_drift(graph, ready, spikes)
    report = RewriteReport(
        spec.name,
        tuple(acc["applied"]),
        tuple(acc["skipped"]),
        tuple(acc["unfixable"]),
        bool(acc["applied"]),
        drift,
    )
    return RewriteResult(ready, report)
