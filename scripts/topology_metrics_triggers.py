"""The four named ARCH-0001 triggers and their clause states."""

from typing import Dict, List, NamedTuple, Tuple

from topology_metrics_clause import Clause
from topology_metrics_metrics import _ge, _or


class Trigger(NamedTuple):
    """A named trigger, its phase, its rule, and its clause states."""

    name: str
    phase: str
    mode: str
    clauses: Tuple[Clause, ...]

    @property
    def fired(self) -> bool:
        """Return whether the trigger's clauses fire it."""
        if self.mode == "any":
            return any(clause.state is True for clause in self.clauses)
        return all(clause.state is True for clause in self.clauses)


def _t1_clauses(metrics: Dict[str, object]) -> Tuple[Clause, ...]:
    """Return T1's clauses: dashboard (Phase 2) evidence."""
    build_slow = _or(
        (
            _ge(metrics["dashboard_build_minutes"], 10.0),
            _ge(metrics["dashboard_build_share"], 0.40),
        )
    )
    return (
        Clause(
            "isolated client-only change sets >= 4",
            _ge(metrics["dashboard_isolated"], 4),
        ),
        Clause("client build >= 10 min or >= 40% wall-clock", build_slow),
        Clause(
            "client PRs blocked on Python review >= 3",
            _ge(metrics["dashboard_blocked_prs"], 3),
        ),
    )


def _trigger_t1(metrics: Dict[str, object]) -> Trigger:
    """Build T1: the dashboard (Phase 2) trigger."""
    return Trigger(
        name="T1",
        phase="dashboard (Phase 2)",
        mode="any",
        clauses=_t1_clauses(metrics),
    )


def _t2_clauses(metrics: Dict[str, object]) -> Tuple[Clause, ...]:
    """Return T2's clauses: spikeforge-targets (Phase 3) evidence."""
    return (
        Clause(
            "backend-SDK-caused core releases >= 2",
            _ge(metrics["backend_sdk_core_releases"], 2),
        ),
        Clause(
            "one backend SDK forces 2 releases <= 30d",
            bool(metrics["backend_sdk_two_in_30"]),
        ),
        Clause(
            "isolated targets change sets >= 6",
            _ge(metrics["targets_isolated"], 6),
        ),
    )


def _trigger_t2(metrics: Dict[str, object]) -> Trigger:
    """Build T2: the spikeforge-targets (Phase 3) trigger."""
    return Trigger(
        name="T2",
        phase="spikeforge-targets (Phase 3)",
        mode="any",
        clauses=_t2_clauses(metrics),
    )


def _t3_clauses(metrics: Dict[str, object]) -> Tuple[Clause, ...]:
    """Return T3's clauses: spikeforge-hub (Phase 4 go) evidence."""
    return (
        Clause(
            "hub-only release demand >= 2",
            _ge(metrics["hub_only_release_demand"], 2),
        ),
        Clause(
            "catalog schema_version unchanged >= 2 releases",
            metrics["catalog_unchanged"],
        ),
        Clause(
            "isolated hub change sets >= 15% of core commits",
            _ge(metrics["hub_share"], 0.15),
        ),
    )


def _trigger_t3(metrics: Dict[str, object]) -> Trigger:
    """Build T3: the spikeforge-hub (Phase 4 go) trigger."""
    return Trigger(
        name="T3",
        phase="spikeforge-hub (Phase 4 go)",
        mode="all",
        clauses=_t3_clauses(metrics),
    )


def _t4_clauses(metrics: Dict[str, object]) -> Tuple[Clause, ...]:
    """Return T4's clauses: spikeforge-server (Phase 4 conditional)."""
    return (
        Clause(
            "server-only release demand >= 3",
            _ge(metrics["server_only_release_demand"], 3),
        ),
        Clause(
            "core pin conflicts >= 2 in window",
            _ge(metrics["pin_conflict_commits"], 2),
        ),
        Clause(
            "isolated server change sets >= 20% of core commits",
            _ge(metrics["server_share"], 0.20),
        ),
    )


def _trigger_t4(metrics: Dict[str, object]) -> Trigger:
    """Build T4: the spikeforge-server (Phase 4 conditional) trigger."""
    return Trigger(
        name="T4",
        phase="spikeforge-server (Phase 4 conditional)",
        mode="all",
        clauses=_t4_clauses(metrics),
    )


def _build_triggers(metrics: Dict[str, object]) -> List[Trigger]:
    """Assemble the four named triggers from the collected metrics."""
    return [
        _trigger_t1(metrics),
        _trigger_t2(metrics),
        _trigger_t3(metrics),
        _trigger_t4(metrics),
    ]

