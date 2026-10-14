"""Derive every ARCH-0001 metric input from collected evidence."""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

from topology_metrics_builds import _dashboard_build_metrics
from topology_metrics_git import _git
from topology_metrics_inputs import _collect_inputs, _Inputs
from topology_metrics_releases import (
    COMPONENTS,
    CommitChanges,
    Release,
    _is_isolated,
)

#: Backend-SDK paths/tokens, used to spot backend-churn-driven releases.
BACKEND_SDK_PREFIX = "spikeforge_targets/backends/"


BACKEND_SDK_TOKENS = ("norse", "lava")


#: Files whose churn proxies a core pin conflict or downgrade.
SERVER_PIN_PATH = "packages/spikeforge-server/pyproject.toml"


#: File whose ``schema_version`` gates the T3(b) clause.
CATALOG_PATH = "spikeforge_hub/models.json"


CORE_PREFIX = "spikeforge/"


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


def _isolated(changes: CommitChanges, component: str) -> int:
    """Count single-component change sets in ``changes``."""
    return sum(
        1 for _, paths in changes if _is_isolated(paths, component)
    )


def _core_commit_count(changes: CommitChanges) -> int:
    """Count commits in ``changes`` that touch the core distribution."""
    return sum(
        1
        for _, paths in changes
        if any(path.startswith(CORE_PREFIX) for path in paths)
    )


def _count_isolated_releases(releases: List[Release], component: str) -> int:
    """Count releases whose changed paths touch only ``component``."""
    return sum(
        1
        for release in releases
        if _is_isolated(release.changed_paths, component)
    )


def _count_pin_conflicts(changes: CommitChanges) -> int:
    """Count commits in ``changes`` that touch the core server pin."""
    return sum(1 for _, paths in changes if SERVER_PIN_PATH in paths)


def _release_metrics(data: _Inputs) -> Dict[str, object]:
    """Return the release- and isolated-change-set count fields."""
    return {
        "releases_in_window": len(data.in_window),
        "core_releases_in_window": len(data.core_window),
        "dashboard_isolated": _isolated(data.window_changes, "dashboard"),
        "targets_isolated": _isolated(data.window_changes, "targets"),
    }


def _backend_demand_metrics(
    in_window: List[Release], backend_releases: List[Release]
) -> Dict[str, object]:
    """Return the backend-SDK and hub/server release-demand fields."""
    return {
        "backend_sdk_core_releases": len(backend_releases),
        "backend_sdk_two_in_30": _two_releases_within(backend_releases, 30),
        "hub_only_release_demand": _count_isolated_releases(
            in_window, "hub"
        ),
        "server_only_release_demand": _count_isolated_releases(
            in_window, "server"
        ),
    }


def _share_metrics(
    window_changes: CommitChanges,
    core_commits: int,
    hub_isolated: int,
    server_isolated: int,
) -> Dict[str, object]:
    """Return the isolated-count, commit-share, and pin-conflict fields."""
    return {
        "hub_isolated": hub_isolated,
        "server_isolated": server_isolated,
        "hub_share": hub_isolated / core_commits if core_commits else 0.0,
        "server_share": (
            server_isolated / core_commits if core_commits else 0.0
        ),
        "pin_conflict_commits": _count_pin_conflicts(window_changes),
    }


def _demand_metrics(data: _Inputs) -> Dict[str, object]:
    """Return the demand/share/backend-SDK fields."""
    core_commits = _core_commit_count(data.quarter_changes)
    hub_isolated = _isolated(data.quarter_changes, "hub")
    server_isolated = _isolated(data.quarter_changes, "server")
    backend_releases = [
        release
        for release in data.core_window
        if _is_backend_sdk_release(release)
    ]
    fields = _backend_demand_metrics(data.in_window, backend_releases)
    fields.update(
        _share_metrics(
            data.window_changes, core_commits, hub_isolated, server_isolated
        )
    )
    return fields


def _build_metrics(data: _Inputs) -> Dict[str, object]:
    """Return the catalog-stability and dashboard-build fields."""
    catalog = _catalog_unchanged(_catalog_versions(data.in_window))
    build = _dashboard_build_metrics()
    return {
        "catalog_unchanged": catalog,
        "dashboard_build_minutes": build.minutes,
        "dashboard_build_share": build.share,
        "dashboard_blocked_prs": build.blocked_prs,
    }


def _collect(as_of: datetime, window_days: int) -> Dict[str, object]:
    """Compute every metric input from git and (best effort) GitHub."""
    data = _collect_inputs(as_of, window_days)
    fields = _release_metrics(data)
    fields.update(_demand_metrics(data))
    fields.update(_build_metrics(data))
    return fields

