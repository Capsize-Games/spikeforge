"""Build named structural mismatches between an artifact and a preset.

These helpers turn a structural difference into one readable record naming the
stage and reason, which is what makes a rejected import explain *why* rather
than reporting a generic failure. Names carry no underscore because they are
this module's small public vocabulary for
:mod:`spikeforge_hub.compat`.
"""

from typing import Any, Dict, List, Mapping, Sequence, Tuple

#: A single named mismatch record.
Record = Dict[str, str]


def short(value: Any, limit: int = 120) -> str:
    """Return ``value`` as text, truncated so a payload stays readable."""
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


def node_index(
    nodes: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Return a node record mapping keyed by node name."""
    return {str(node["name"]): dict(node) for node in nodes}


def edge_set(edges: Sequence[Mapping[str, Any]]) -> set:
    """Return the ``(source, target)`` pairs of an edge listing."""
    return {(str(edge["source"]), str(edge["target"])) for edge in edges}


def record(stage: str, reason: str, expected: str, actual: str) -> Record:
    """Build one JSON-able mismatch record."""
    return {
        "stage": str(stage),
        "reason": reason,
        "expected": str(expected),
        "actual": str(actual),
    }


def node_mismatches(
    actual: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
) -> List[Record]:
    """Return one named mismatch record per differing node."""
    found: List[Record] = []
    for name, node in expected.items():
        seen = actual.get(name)
        if seen is None:
            found.append(record(name, "missing", node["kind"], "absent"))
        elif seen["kind"] != node["kind"]:
            found.append(record(name, "kind", node["kind"], seen["kind"]))
        elif seen["params"] != node["params"]:
            found.append(record(
                name, "params", short(node["params"]), short(seen["params"])
            ))
    for name, node in actual.items():
        if name not in expected:
            found.append(record(name, "unexpected", "absent", node["kind"]))
    return found


def edge_mismatches(actual: set, expected: set) -> List[Record]:
    """Return one named mismatch record per differing edge."""
    found = [
        record(source, "edge-missing", target, "absent")
        for source, target in sorted(expected - actual)
    ]
    found.extend(
        record(source, "edge-unexpected", "absent", target)
        for source, target in sorted(actual - expected)
    )
    return found


def stage_of(key: str) -> str:
    """Return the stage prefix of a state-dict key."""
    return key.split(".", 1)[0]


def _unexpected_keys(
    actual: Mapping[str, Tuple[int, ...]],
    expected: Mapping[str, Tuple[int, ...]],
) -> List[Record]:
    """Return an unexpected-key record per key absent from the preset."""
    return [
        record(stage_of(key), "unexpected-key", "absent", key)
        for key in actual
        if key not in expected
    ]


def state_mismatches(
    actual: Mapping[str, Tuple[int, ...]],
    expected: Mapping[str, Tuple[int, ...]],
) -> List[Record]:
    """Return one named mismatch record per differing state-dict key."""
    found: List[Record] = []
    for key, shape in expected.items():
        seen = actual.get(key)
        if seen is None:
            found.append(record(
                stage_of(key), "missing-key", str(list(shape)), "absent"
            ))
        elif seen != shape:
            found.append(record(
                stage_of(key), "shape", str(list(shape)), str(list(seen))
            ))
    found.extend(_unexpected_keys(actual, expected))
    return found
