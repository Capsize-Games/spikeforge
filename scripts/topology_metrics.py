#!/usr/bin/env python3
"""Compute the ARCH-0001 topology triggers (T1-T4) from local evidence.

See ``plans/arch-0001-decision-metrics.md`` for the trigger definitions. The
script is deliberately stdlib-first and read-only: it derives what it can from
``git log`` and the pushed ``*-v*`` tags, and augments the picture with GitHub
Actions timing through the ``gh`` CLI when that is available.

Every input degrades to ``None`` (reported as "unavailable") rather than
raising, so the script runs offline, in a shallow clone, and in CI. A trigger
with an unavailable input never fires; the printed status names the missing
input so the maintainer can record the decision in the ADR.

Usage::

    python scripts/topology_metrics.py [--window-days 90] [--as-of DATE]

It prints the metric values, then, for each named trigger, whether it fired and
which clauses held. ``--json`` emits the same data as JSON.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

#: Repository root (this file lives in ``scripts/``).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Trailing rolling window used for the "releases" clause of every trigger.
DEFAULT_WINDOW_DAYS = 90

#: Approximate quarter, used for the "change-set share" clauses.
QUARTER_DAYS = 91

#: The core distribution owns this tag prefix and this import root.
CORE_DISTRIBUTION = "spikeforge"
CORE_PREFIX = "spikeforge/"

#: Component prefixes from ``plans/arch-0001-decision-metrics.md``.
COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "dashboard": ("client/",),
    "targets": ("spikeforge_targets/",),
    "hub": ("spikeforge_hub/",),
    "server": ("server/",),
}

#: Backend-SDK paths/tokens, used to spot backend-churn-driven releases.
BACKEND_SDK_PREFIX = "spikeforge_targets/backends/"
BACKEND_SDK_TOKENS = ("norse", "lava")

#: Files whose churn proxies a core pin conflict or downgrade.
SERVER_PIN_PATH = "packages/spikeforge-server/pyproject.toml"

#: File whose ``schema_version`` gates the T3(b) clause.
CATALOG_PATH = "spikeforge_hub/models.json"

#: Release tags are ``<distribution>-v<version>``.
TAG_RE = re.compile(r"^(?P<dist>.+)-v(?P<version>\d+\.\d+.*)$")

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


class Release(NamedTuple):
    """A pushed release tag and the git evidence derived from it."""

    tag: str
    distribution: str
    version: str
    date: datetime
    changed_paths: Tuple[str, ...]
    messages: str


class DashboardBuild(NamedTuple):
    """Best-effort client build timing scraped from the Actions API."""

    minutes: Optional[float]
    share: Optional[float]
    blocked_prs: Optional[int]


class Clause(NamedTuple):
    """One trigger clause and its evaluated state (None == unavailable)."""

    text: str
    state: Optional[bool]


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


def _run(command: Sequence[str]) -> Optional[str]:
    """Run ``command`` in the repo and return stdout, or None on failure."""
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _git(*args: str) -> Optional[str]:
    """Run ``git`` with ``args`` and return stdout, or None."""
    return _run(("git", *args))


def _gh_json(args: Sequence[str]) -> Optional[object]:
    """Return parsed ``gh`` JSON output, or None when it is unavailable."""
    output = _run(("gh", *args))
    if output is None:
        return None
    try:
        return json.loads(output)
    except ValueError:
        return None


def _parse_tag(tag: str) -> Optional[Tuple[str, str]]:
    """Split ``<distribution>-v<version>`` into its two parts."""
    match = TAG_RE.match(tag)
    if match is None:
        return None
    return match.group("dist"), match.group("version")


def _parse_stamp(stamp: str) -> Optional[datetime]:
    """Parse an ISO-8601 git timestamp into an aware datetime."""
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _tags() -> List[Tuple[str, datetime]]:
    """Return ``(tag, committed date)`` for every release tag, newest first."""
    output = _git(
        "for-each-ref",
        "--sort=-creatordate",
        "--format=%(refname:short) %(creatordate:iso-strict)",
        "refs/tags",
    )
    if not output:
        return []
    tags: List[Tuple[str, datetime]] = []
    for line in output.splitlines():
        tag, _, stamp = line.partition(" ")
        if _parse_tag(tag) is None or not stamp.strip():
            continue
        date = _parse_stamp(stamp.strip())
        if date is not None:
            tags.append((tag, date))
    return tags


def _changed_paths(rev_range: str) -> Tuple[str, ...]:
    """Return the files changed by ``rev_range`` (empty when unknown)."""
    output = _git("diff", "--name-only", rev_range)
    if not output:
        return ()
    return tuple(line for line in output.splitlines() if line.strip())


def _releases() -> List[Release]:
    """Build release records from tags, oldest range first."""
    ordered = list(reversed(_tags()))
    releases: List[Release] = []
    previous: Optional[str] = None
    for tag, date in ordered:
        parsed = _parse_tag(tag)
        if parsed is None:
            continue
        rev_range = f"{previous}..{tag}" if previous else tag
        releases.append(
            Release(
                tag=tag,
                distribution=parsed[0],
                version=parsed[1],
                date=date,
                changed_paths=_changed_paths(rev_range),
                messages=_git("log", "--format=%s%n%b", rev_range) or "",
            )
        )
        previous = tag
    return releases


def _components_for(paths: Sequence[str]) -> List[str]:
    """Return the components every path in ``paths`` falls inside."""
    if not paths:
        return []
    return [
        component
        for component, prefixes in COMPONENTS.items()
        if all(path.startswith(prefixes) for path in paths)
    ]


def _is_isolated(paths: Sequence[str], component: str) -> bool:
    """Return True for a non-empty, single-component change set."""
    return _components_for(paths) == [component]


def _commit_changes(
    since: datetime,
) -> List[Tuple[str, Tuple[str, ...]]]:
    """Return ``(sha, paths)`` for non-merge commits since ``since``."""
    stamp = since.strftime("%Y-%m-%d %H:%M:%S")
    output = _git(
        "log",
        f"--since={stamp}",
        "--no-merges",
        "--name-only",
        "--pretty=format:__COMMIT__%H",
    )
    if not output:
        return []
    changes: List[Tuple[str, Tuple[str, ...]]] = []
    sha: Optional[str] = None
    paths: List[str] = []

    def flush() -> None:
        """Append the pending commit record, if any."""
        if sha is not None:
            changes.append((sha, tuple(paths)))

    for line in output.splitlines():
        if line.startswith("__COMMIT__"):
            flush()
            sha = line[len("__COMMIT__") :]
            paths = []
        elif line.strip():
            paths.append(line)
    flush()
    return changes


def _is_backend_sdk_release(release: Release) -> bool:
    """Return True when a core release was caused only by a backend SDK."""
    paths = release.changed_paths
    if not paths:
        return False
    if not all(path.startswith(COMPONENTS["targets"]) for path in paths):
        return False
    lowered = release.messages.lower()
    return any(token in lowered for token in BACKEND_SDK_TOKENS)


def _two_releases_within(
    releases: Sequence[Release], days: int
) -> bool:
    """Return True when two releases are ``days`` or fewer apart."""
    ordered = sorted(releases, key=lambda release: release.date)
    return any(
        later.date - earlier.date <= timedelta(days=days)
        for earlier, later in zip(ordered, ordered[1:])
    )


def _schema_version(text: Optional[str]) -> Optional[str]:
    """Return the ``schema_version`` in a catalog document, or None."""
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("schema_version")
    return None if value is None else str(value)


def _catalog_versions(
    releases: Sequence[Release],
) -> List[Optional[str]]:
    """Return the catalog ``schema_version`` captured at each release."""
    return [
        _schema_version(_git("show", f"{release.tag}:{CATALOG_PATH}"))
        for release in releases
    ]


def _catalog_unchanged(
    versions: Sequence[Optional[str]],
) -> Optional[bool]:
    """Return whether the schema version held across two or more releases."""
    known = [version for version in versions if version is not None]
    if len(known) < 2:
        return None
    return len(set(known)) == 1


def _duration_seconds(start: object, end: object) -> float:
    """Return the seconds between two ISO timestamps, or 0.0."""
    if not isinstance(start, str) or not isinstance(end, str):
        return 0.0
    first = _parse_stamp(start)
    second = _parse_stamp(end)
    if first is None or second is None:
        return 0.0
    return max(0.0, (second - first).total_seconds())


def _job_seconds(jobs: object) -> Tuple[float, float]:
    """Return ``(client seconds, total seconds)`` from a jobs payload."""
    if not isinstance(jobs, dict):
        return 0.0, 0.0
    entries = jobs.get("jobs")
    if not isinstance(entries, list):
        return 0.0, 0.0
    client = 0.0
    total = 0.0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        duration = _duration_seconds(
            entry.get("started_at"), entry.get("completed_at")
        )
        total += duration
        if entry.get("name") == "client":
            client += duration
    return client, total


def _dashboard_build_metrics() -> DashboardBuild:
    """Best-effort client build timing from the GitHub Actions API."""
    repo = _gh_json(("repo", "view", "--json", "nameWithOwner"))
    slug = repo.get("nameWithOwner") if isinstance(repo, dict) else None
    runs = _gh_json(
        (
            "run",
            "list",
            "--workflow",
            "ci.yml",
            "--status",
            "completed",
            "--limit",
            "10",
            "--json",
            "databaseId,conclusion",
        )
    )
    if not isinstance(slug, str) or not isinstance(runs, list):
        return DashboardBuild(None, None, None)
    minutes: List[float] = []
    shares: List[float] = []
    for run in runs:
        run_id = run.get("databaseId") if isinstance(run, dict) else None
        if run_id is None:
            continue
        jobs = _gh_json(
            ("api", f"repos/{slug}/actions/runs/{run_id}/jobs")
        )
        client, total = _job_seconds(jobs)
        if total:
            minutes.append(client / 60.0)
            shares.append(client / total)
    if not minutes:
        return DashboardBuild(None, None, None)
    return DashboardBuild(
        sum(minutes) / len(minutes),
        sum(shares) / len(shares),
        None,
    )


def _ge(value: object, threshold: float) -> Optional[bool]:
    """Return ``value >= threshold``, or None when the value is missing."""
    if value is None:
        return None
    return float(value) >= threshold


def _or(states: Sequence[Optional[bool]]) -> Optional[bool]:
    """Combine clause states with OR, preserving an unknown result."""
    if any(state is True for state in states):
        return True
    if any(state is None for state in states):
        return None
    return False


def _collect(as_of: datetime, window_days: int) -> Dict[str, object]:
    """Compute every metric input from git and (best effort) GitHub."""
    releases = _releases()
    window_start = as_of - timedelta(days=window_days)
    quarter_start = as_of - timedelta(days=QUARTER_DAYS)

    in_window = [
        release for release in releases if release.date >= window_start
    ]
    core_window = [
        release
        for release in in_window
        if release.distribution == CORE_DISTRIBUTION
    ]
    window_changes = _commit_changes(window_start)
    quarter_changes = _commit_changes(quarter_start)

    def isolated(
        changes: Sequence[Tuple[str, Tuple[str, ...]]], component: str
    ) -> int:
        """Count single-component change sets in ``changes``."""
        return sum(
            1 for _, paths in changes if _is_isolated(paths, component)
        )

    core_commits = sum(
        1
        for _, paths in quarter_changes
        if any(path.startswith(CORE_PREFIX) for path in paths)
    )
    hub_isolated = isolated(quarter_changes, "hub")
    server_isolated = isolated(quarter_changes, "server")

    backend_releases = [
        release for release in core_window if _is_backend_sdk_release(release)
    ]
    catalog = _catalog_unchanged(_catalog_versions(in_window))
    build = _dashboard_build_metrics()

    return {
        "releases_in_window": len(in_window),
        "core_releases_in_window": len(core_window),
        "dashboard_isolated": isolated(window_changes, "dashboard"),
        "targets_isolated": isolated(window_changes, "targets"),
        "hub_isolated": hub_isolated,
        "server_isolated": server_isolated,
        "backend_sdk_core_releases": len(backend_releases),
        "backend_sdk_two_in_30": _two_releases_within(backend_releases, 30),
        "hub_only_release_demand": sum(
            1
            for release in in_window
            if _is_isolated(release.changed_paths, "hub")
        ),
        "server_only_release_demand": sum(
            1
            for release in in_window
            if _is_isolated(release.changed_paths, "server")
        ),
        "hub_share": hub_isolated / core_commits if core_commits else 0.0,
        "server_share": (
            server_isolated / core_commits if core_commits else 0.0
        ),
        "pin_conflict_commits": sum(
            1
            for _, paths in window_changes
            if SERVER_PIN_PATH in paths
        ),
        "catalog_unchanged": catalog,
        "dashboard_build_minutes": build.minutes,
        "dashboard_build_share": build.share,
        "dashboard_blocked_prs": build.blocked_prs,
    }


def _build_triggers(metrics: Dict[str, object]) -> List[Trigger]:
    """Assemble the four named triggers from the collected metrics."""
    return [
        Trigger(
            name="T1",
            phase="dashboard (Phase 2)",
            mode="any",
            clauses=(
                Clause(
                    "isolated client-only change sets >= 4",
                    _ge(metrics["dashboard_isolated"], 4),
                ),
                Clause(
                    "client build >= 10 min or >= 40% wall-clock",
                    _or(
                        (
                            _ge(metrics["dashboard_build_minutes"], 10.0),
                            _ge(metrics["dashboard_build_share"], 0.40),
                        )
                    ),
                ),
                Clause(
                    "client PRs blocked on Python review >= 3",
                    _ge(metrics["dashboard_blocked_prs"], 3),
                ),
            ),
        ),
        Trigger(
            name="T2",
            phase="spikeforge-targets (Phase 3)",
            mode="any",
            clauses=(
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
            ),
        ),
        Trigger(
            name="T3",
            phase="spikeforge-hub (Phase 4 go)",
            mode="all",
            clauses=(
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
            ),
        ),
        Trigger(
            name="T4",
            phase="spikeforge-server (Phase 4 conditional)",
            mode="all",
            clauses=(
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
            ),
        ),
    ]


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
        "triggers": [
            {
                "name": trigger.name,
                "phase": trigger.phase,
                "mode": trigger.mode,
                "fired": trigger.fired,
                "clauses": [
                    {"text": clause.text, "state": clause.state}
                    for clause in trigger.clauses
                ],
            }
            for trigger in triggers
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_as_of(value: Optional[str]) -> datetime:
    """Return the window anchor, defaulting to the current UTC time."""
    if not value:
        return datetime.now(timezone.utc)
    parsed = _parse_stamp(value)
    if parsed is None:
        raise SystemExit(f"invalid --as-of value: {value!r}")
    return parsed


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute the ARCH-0001 topology triggers T1-T4."
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"rolling window in days (default: {DEFAULT_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="window anchor as an ISO date (default: now, UTC)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the metrics and statuses as JSON",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Compute and print the metrics and trigger statuses."""
    args = _parse_args(argv)
    as_of = _parse_as_of(args.as_of)
    metrics = _collect(as_of, args.window_days)
    triggers = _build_triggers(metrics)
    if args.json:
        _print_json(metrics, triggers, args.window_days, as_of)
    else:
        _print_report(metrics, triggers, args.window_days, as_of)
    return 0


if __name__ == "__main__":
    sys.exit(main())
