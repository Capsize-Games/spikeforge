"""Collect release tags and commit change sets from git."""

from datetime import datetime
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from topology_metrics_git import (
    _changed_paths,
    _git,
    _parse_tag,
    _tags,
)

#: Component prefixes from ``plans/arch-0001-decision-metrics.md``.
COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "dashboard": ("client/",),
    "targets": ("spikeforge_targets/",),
    "hub": ("spikeforge_hub/",),
    "server": ("server/",),
}


#: One non-merge commit's sha and the paths it touched.
CommitChanges = Sequence[Tuple[str, Tuple[str, ...]]]


class Release(NamedTuple):
    """A pushed release tag and the git evidence derived from it."""

    tag: str
    distribution: str
    version: str
    date: datetime
    changed_paths: Tuple[str, ...]
    messages: str


def _release_record(
    tag: str,
    date: datetime,
    parsed: Tuple[str, str],
    previous: Optional[str],
) -> Release:
    """Build the release record for ``tag`` against ``previous``."""
    rev_range = f"{previous}..{tag}" if previous else tag
    return Release(
        tag=tag,
        distribution=parsed[0],
        version=parsed[1],
        date=date,
        changed_paths=_changed_paths(rev_range),
        messages=_git("log", "--format=%s%n%b", rev_range) or "",
    )


def _releases() -> List[Release]:
    """Build release records from tags, oldest range first."""
    ordered = list(reversed(_tags()))
    releases: List[Release] = []
    previous: Optional[str] = None
    for tag, date in ordered:
        parsed = _parse_tag(tag)
        if parsed is None:
            continue
        releases.append(_release_record(tag, date, parsed, previous))
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


def _git_log(since: datetime) -> Optional[str]:
    """Return the non-merge log with names since ``since``."""
    stamp = since.strftime("%Y-%m-%d %H:%M:%S")
    return _git(
        "log",
        f"--since={stamp}",
        "--no-merges",
        "--name-only",
        "--pretty=format:__COMMIT__%H",
    )


def _flush(
    changes: List[Tuple[str, Tuple[str, ...]]],
    sha: Optional[str],
    paths: List[str],
) -> None:
    """Append the pending ``(sha, paths)`` record, if any."""
    if sha is not None:
        changes.append((sha, tuple(paths)))


def _parse_log(output: str) -> CommitChanges:
    """Parse ``(sha, paths)`` records out of a name-only log."""
    changes: List[Tuple[str, Tuple[str, ...]]] = []
    sha: Optional[str] = None
    paths: List[str] = []
    for line in output.splitlines():
        if line.startswith("__COMMIT__"):
            _flush(changes, sha, paths)
            sha = line[len("__COMMIT__") :]
            paths = []
        elif line.strip():
            paths.append(line)
    _flush(changes, sha, paths)
    return changes


def _commit_changes(since: datetime) -> CommitChanges:
    """Return ``(sha, paths)`` for non-merge commits since ``since``."""
    output = _git_log(since)
    if not output:
        return []
    return _parse_log(output)

