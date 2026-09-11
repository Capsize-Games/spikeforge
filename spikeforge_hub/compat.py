"""Map an inspected artifact to a shipped preset, or reject it by name.

Gate two of the import funnel. The inspected structure is compared against the
presets declared in :mod:`spikeforge.topology.registry`:

* ``exact`` — a NIR graph whose nodes, kinds, parameter shapes, and edges match
  a preset's rendering exactly.
* ``mappable`` — a state dict whose keys and tensor shapes match a preset
  module exactly, so its weights can be loaded under a stage mapping.
* ``incompatible`` — every candidate is reported with the specific stages that
  differ, so a mismatch is always named rather than silently loaded.
"""

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from spikeforge_hub.artifact_report import (
    NIR_GRAPH,
    STATE_DICT,
    ArtifactReport,
    normalize_nodes,
)
from spikeforge_hub.expected import (
    candidates,
    expected_graph,
    expected_state,
)
from spikeforge_hub.mismatches import (
    Record,
    edge_mismatches,
    edge_set,
    node_index,
    node_mismatches,
    stage_of,
    state_mismatches,
)

#: The artifact matches a preset exactly and may be promoted.
EXACT = "exact"
#: The artifact's weights map onto a preset and may be promoted.
MAPPABLE = "mappable"
#: No preset matches; the artifact is rejected with named mismatches.
INCOMPATIBLE = "incompatible"

_Diff = Callable[[str], Optional[List[Record]]]
_Resolution = Tuple[str, List[Record]]


@dataclass(frozen=True)
class CompatibilityVerdict:
    """Gate two's result: a preset match, or the mismatches that reject it."""

    verdict: str
    topology: Optional[str]
    mapping: Dict[str, str]
    mismatches: Tuple[Dict[str, str], ...]
    notes: Tuple[str, ...]

    @property
    def compatible(self) -> bool:
        """Return True when the artifact may be promoted into the store."""
        return self.verdict in (EXACT, MAPPABLE)

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-able verdict the socket and CLI emit."""
        return {
            "verdict": self.verdict,
            "topology": self.topology,
            "mapping": dict(self.mapping),
            "mismatches": [dict(item) for item in self.mismatches],
            "notes": list(self.notes),
        }


def _graph_diff(report: ArtifactReport, name: str) -> Optional[List[Record]]:
    """Return the structural differences between a graph and one preset."""
    expected = expected_graph(name)
    if expected is None:
        return None
    want = normalize_nodes(expected["nodes"])
    found = node_mismatches(node_index(report.nodes), node_index(want))
    found.extend(
        edge_mismatches(edge_set(report.edges), edge_set(expected["edges"]))
    )
    return found


def _graph_mapping(report: ArtifactReport) -> Dict[str, str]:
    """Return the identity stage mapping for an exact NIR match."""
    return {str(node["name"]): str(node["name"]) for node in report.nodes}


def _state_diff(
    actual: Mapping[str, Tuple[int, ...]], name: str
) -> Optional[List[Record]]:
    """Return the key differences between a state dict and one preset."""
    expected = expected_state(name)
    if expected is None:
        return None
    return state_mismatches(actual, expected)


def _state_mapping(actual: Mapping[str, Tuple[int, ...]]) -> Dict[str, str]:
    """Return the ``key -> stage`` mapping for a mappable state dict."""
    return {key: stage_of(key) for key in actual}


def _resolve(names: Sequence[str], diff: _Diff) -> Optional[_Resolution]:
    """Return the first exact match, else the closest candidate's diff."""
    best: Optional[Tuple[int, str, List[Record]]] = None
    for name in names:
        mismatch = diff(name)
        if mismatch is None:
            continue
        if not mismatch:
            return name, mismatch
        if best is None or len(mismatch) < best[0]:
            best = (len(mismatch), name, mismatch)
    return None if best is None else (best[1], best[2])


def _incompatible(topology: Optional[str], note: str) -> CompatibilityVerdict:
    """Return an incompatible verdict carrying a single explanatory note."""
    return CompatibilityVerdict(INCOMPATIBLE, topology, {}, (), (note,))


def _closest(name: str, mismatch: List[Record]) -> CompatibilityVerdict:
    """Return an incompatible verdict naming the closest preset's diff."""
    note = f"closest preset {name!r} differs in {len(mismatch)} place(s)"
    return CompatibilityVerdict(
        INCOMPATIBLE, name, {}, tuple(mismatch), (note,)
    )


def _classify_graph(
    report: ArtifactReport, topology: Optional[str]
) -> CompatibilityVerdict:
    """Compare a NIR graph against the candidate presets."""
    result = _resolve(
        candidates(topology), lambda name: _graph_diff(report, name)
    )
    if result is None:
        return _incompatible(None, "no comparable preset for this graph")
    name, mismatch = result
    if not mismatch:
        return CompatibilityVerdict(
            EXACT, name, _graph_mapping(report), (), ()
        )
    return _closest(name, mismatch)


def _classify_state(
    report: ArtifactReport, topology: Optional[str]
) -> CompatibilityVerdict:
    """Compare a torch state dict against the candidate presets."""
    actual = {str(key["key"]): tuple(key["shape"]) for key in report.keys}
    if not actual:
        return _incompatible(None, "state dict declares no keys")
    result = _resolve(
        candidates(topology), lambda name: _state_diff(actual, name)
    )
    if result is None:
        return _incompatible(None, "no comparable preset for this state dict")
    name, mismatch = result
    if not mismatch:
        return CompatibilityVerdict(
            MAPPABLE, name, _state_mapping(actual), (), ()
        )
    return _closest(name, mismatch)


def classify(
    report: ArtifactReport, topology: Optional[str] = None
) -> CompatibilityVerdict:
    """Return the compatibility verdict for an inspected ``report``."""
    if report.kind == NIR_GRAPH:
        return _classify_graph(report, topology)
    if report.kind == STATE_DICT:
        return _classify_state(report, topology)
    return _incompatible(None, f"{report.kind} has no structural mapping")
