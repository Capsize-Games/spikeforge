"""Best-effort client build timing from the Actions API."""

from typing import List, NamedTuple, Optional, Tuple

from topology_metrics_git import _gh_json, _parse_stamp


class DashboardBuild(NamedTuple):
    """Best-effort client build timing scraped from the Actions API."""

    minutes: Optional[float]
    share: Optional[float]
    blocked_prs: Optional[int]


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


def _completed_runs() -> Optional[Tuple[str, list]]:
    """Return the repo slug and raw completed runs, if available."""
    repo = _gh_json(("repo", "view", "--json", "nameWithOwner"))
    slug = repo.get("nameWithOwner") if isinstance(repo, dict) else None
    runs = _gh_json(
        (
            "run", "list", "--workflow", "ci.yml", "--status",
            "completed", "--limit", "10", "--json",
            "databaseId,conclusion",
        )
    )
    if not isinstance(slug, str) or not isinstance(runs, list):
        return None
    return slug, runs


def _completed_run_ids() -> Optional[Tuple[str, List[int]]]:
    """Return the repo slug and recent completed CI run ids."""
    found = _completed_runs()
    if found is None:
        return None
    slug, runs = found
    ids = [
        run.get("databaseId")
        for run in runs
        if isinstance(run, dict)
        and run.get("databaseId") is not None
    ]
    return slug, ids


def _run_share(slug: str, run_id: int) -> Optional[Tuple[float, float]]:
    """Return ``(client minutes, client share)`` for one run."""
    jobs = _gh_json(
        ("api", f"repos/{slug}/actions/runs/{run_id}/jobs")
    )
    client, total = _job_seconds(jobs)
    if not total:
        return None
    return client / 60.0, client / total


def _average(values: List[float]) -> float:
    """Return the mean of ``values``."""
    return sum(values) / len(values)


def _dashboard_build_metrics() -> DashboardBuild:
    """Best-effort client build timing from the GitHub Actions API."""
    found = _completed_run_ids()
    if found is None:
        return DashboardBuild(None, None, None)
    slug, run_ids = found
    minutes: List[float] = []
    shares: List[float] = []
    for run_id in run_ids:
        share = _run_share(slug, run_id)
        if share is None:
            continue
        minutes.append(share[0])
        shares.append(share[1])
    if not minutes:
        return DashboardBuild(None, None, None)
    return DashboardBuild(_average(minutes), _average(shares), None)

