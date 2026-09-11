"""Compare two benchmark records and flag regressions past a threshold.

The comparator matches records on ``(topology, mode)`` and reports the
relative change of the throughput and memory metrics. A candidate regresses
when a throughput metric moves the wrong way by more than the threshold, or
when its peak memory grows by more than it. The result is JSON-able, and
:func:`exit_code` turns it into a CI gate.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

#: Default relative change (fraction) that counts as a regression.
DEFAULT_THRESHOLD = 0.10

#: (reported metric, direction) where "lower" means a larger value is worse.
_METRICS: Tuple[Tuple[str, str], ...] = (
    ("ms_per_step", "lower"),
    ("steps_per_second", "higher"),
    ("memory_bytes", "lower"),
)

#: Memory keys probed in order; the first non-zero one is compared.
_MEMORY_KEYS = (
    "tracemalloc_peak_bytes",
    "process_rss_bytes",
    "cuda_peak_bytes",
)


def _case_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    """Return the ``(topology, mode)`` key a record is matched on."""
    return str(record.get("topology")), str(record.get("mode"))


def _memory(record: Mapping[str, Any]) -> Optional[float]:
    """Return the first non-zero memory metric for a record, or None."""
    block = record.get("memory") or {}
    for key in _MEMORY_KEYS:
        value = block.get(key)
        if value:
            return float(value)
    return None


def _value(record: Mapping[str, Any], metric: str) -> Optional[float]:
    """Return the numeric value of ``metric`` for a record, or None."""
    if metric == "memory_bytes":
        return _memory(record)
    key = "mean_ms_per_step" if metric == "ms_per_step" else metric
    value = (record.get("forward") or {}).get(key)
    return None if value is None else float(value)


def _change(
    candidate: Optional[float], baseline: Optional[float]
) -> Optional[float]:
    """Return the relative change from baseline to candidate, or None."""
    if not baseline or candidate is None or baseline is None:
        return None
    return (candidate - baseline) / baseline


def _regressed(
    direction: str, fraction: Optional[float], threshold: float
) -> bool:
    """Return True when ``fraction`` breaches ``threshold`` the wrong way."""
    if fraction is None:
        return False
    if direction == "lower":
        return fraction > threshold
    return fraction < -threshold


def _entry(
    metric: str,
    direction: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    """Return one metric comparison entry."""
    base_value = _value(baseline, metric)
    cand_value = _value(candidate, metric)
    fraction = _change(cand_value, base_value)
    return {
        "metric": metric,
        "baseline": base_value,
        "candidate": cand_value,
        "change_fraction": fraction,
        "regressed": _regressed(direction, fraction, threshold),
    }


def _case(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    """Compare one matched topology/mode pair."""
    entries = [
        _entry(metric, direction, baseline, candidate, threshold)
        for metric, direction in _METRICS
    ]
    return {
        "topology": baseline.get("topology"),
        "mode": baseline.get("mode"),
        "metrics": entries,
        "regressed": any(entry["regressed"] for entry in entries),
    }


def _identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the identifying fields of a run."""
    return {
        "run_id": record.get("run_id"),
        "label": record.get("label"),
        "created_at": record.get("created_at"),
    }


def _name(case: Mapping[str, Any]) -> str:
    """Return a ``topology:mode`` label for a regressed case."""
    return f"{case['topology']}:{case['mode']}"


def _pairs(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> List[Any]:
    """Return matched ``(baseline, candidate)`` records in baseline order."""
    base_map = {_case_key(r): r for r in baseline.get("results", [])}
    cand_map = {_case_key(r): r for r in candidate.get("results", [])}
    return [
        (base_map[key], cand_map[key])
        for key in base_map
        if key in cand_map
    ]


def _result(
    cases: List[Dict[str, Any]],
    threshold: float,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble the JSON-able comparison result."""
    regressions: List[str] = [
        _name(case) for case in cases if case["regressed"]
    ]
    return {
        "threshold": float(threshold),
        "compared": len(cases),
        "cases": cases,
        "regressions": regressions,
        "regressed": bool(regressions),
        "baseline": _identity(baseline),
        "candidate": _identity(candidate),
    }


def compare_runs(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Return per-config deltas and whether ``candidate`` regressed.

    ``baseline`` and ``candidate`` are reports from
    :func:`~spikeforge.benchmark.harness.run_benchmark` (or a stored
    run). Only configs present in both are compared.
    """
    cases = [
        _case(base, cand, threshold)
        for base, cand in _pairs(baseline, candidate)
    ]
    return _result(cases, threshold, baseline, candidate)


def exit_code(result: Mapping[str, Any]) -> int:
    """Return the process exit status: 1 on regression, else 0."""
    return 1 if result.get("regressed") else 0
