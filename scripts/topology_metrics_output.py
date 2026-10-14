"""Print the metrics table and trigger statuses, as text or JSON."""

import json
from datetime import datetime
from typing import Dict, Sequence, Tuple

from topology_metrics_triggers import Trigger

#: Human-readable metric rows, in print order.
_METRIC_ROWS: Tuple[Tuple[str, str], ...] = (
    ("Releases in window", "releases_in_window"),
    ("Core releases in window", "core_releases_in_window"),
    ("Isolated dashboard change sets", "dashboard_isolated"),
    ("Isolated targets change sets", "targets_isolated"),
    ("Isolated hub change sets", "hub_isolated"),
    ("Isolated server change sets", "server_isolated"),
    ("Backend-SDK-caused core releases", "backend_sdk_core_releases"),
    ("Two backend-SDK releases <=30d", "backend_sdk_two_in_30"),
    ("Hub-only release demand", "hub_only_release_demand"),
    ("Server-only release demand", "server_only_release_demand"),
    ("Hub share of core commits (quarter)", "hub_share"),
    ("Server share of core commits (quarter)", "server_share"),
    ("Core pin-conflict commits", "pin_conflict_commits"),
    ("Catalog schema unchanged (>=2)", "catalog_unchanged"),
    ("Dashboard build minutes (avg)", "dashboard_build_minutes"),
    ("Dashboard build share (avg)", "dashboard_build_share"),
    ("Dashboard PRs blocked on review", "dashboard_blocked_prs"),
)


def _format(value: object) -> str:
    """Render a metric value for the human-readable table."""
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _print_report(
    metrics: Dict[str, object],
    triggers: Sequence[Trigger],
    window_days: int,
    as_of: datetime,
) -> None:
    """Print the metric table and the T1-T4 status."""
    stamp = as_of.strftime("%Y-%m-%d %H:%M UTC")
    print(f"ARCH-0001 topology metrics (trailing {window_days}d, {stamp})")
    print("=" * 66)
    for label, key in _METRIC_ROWS:
        print(f"  {label:<44} {_format(metrics[key])}")
    print("-" * 66)
    for trigger in triggers:
        status = "FIRED" if trigger.fired else "not fired"
        print(f"  {trigger.name} {trigger.phase}: {status}")
        for clause in trigger.clauses:
            mark = _format(clause.state)
            print(f"      [{mark:>11}] {clause.text}")


def _trigger_json(trigger: Trigger) -> Dict[str, object]:
    """Return the JSON payload for one trigger status."""
    return {
        "name": trigger.name,
        "phase": trigger.phase,
        "mode": trigger.mode,
        "fired": trigger.fired,
        "clauses": [
            {"text": clause.text, "state": clause.state}
            for clause in trigger.clauses
        ],
    }


def _print_json(
    metrics: Dict[str, object],
    triggers: Sequence[Trigger],
    window_days: int,
    as_of: datetime,
) -> None:
    """Emit the metrics and trigger statuses as JSON."""
    payload = {
        "as_of": as_of.isoformat(),
        "window_days": window_days,
        "metrics": metrics,
        "triggers": [_trigger_json(trigger) for trigger in triggers],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

