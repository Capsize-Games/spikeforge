"""Gather the raw release/commit evidence metrics derive from."""

from datetime import datetime, timedelta
from typing import List, NamedTuple, Tuple

from topology_metrics_releases import (
    CommitChanges,
    Release,
    _commit_changes,
    _releases,
)

#: The core distribution owns this tag prefix and this import root.
CORE_DISTRIBUTION = "spikeforge"


#: Approximate quarter, used for the "change-set share" clauses.
QUARTER_DAYS = 91


class _Inputs(NamedTuple):
    """Collected raw release/commit evidence, before metric derivation."""

    in_window: List[Release]
    core_window: List[Release]
    window_changes: CommitChanges
    quarter_changes: CommitChanges


def _releases_in_window(
    window_start: datetime,
) -> Tuple[List[Release], List[Release]]:
    """Return ``(in_window, core_window)`` releases since ``window_start``."""
    releases = _releases()
    in_window = [
        release for release in releases if release.date >= window_start
    ]
    core_window = [
        release
        for release in in_window
        if release.distribution == CORE_DISTRIBUTION
    ]
    return in_window, core_window


def _collect_inputs(as_of: datetime, window_days: int) -> _Inputs:
    """Gather the raw release/commit evidence ``_collect`` derives from."""
    window_start = as_of - timedelta(days=window_days)
    quarter_start = as_of - timedelta(days=QUARTER_DAYS)
    in_window, core_window = _releases_in_window(window_start)
    return _Inputs(
        in_window=in_window,
        core_window=core_window,
        window_changes=_commit_changes(window_start),
        quarter_changes=_commit_changes(quarter_start),
    )

