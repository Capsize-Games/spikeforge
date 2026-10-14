"""Coordinated-minor-bump check over the compatibility matrix.

A core MINOR bump breaks the satellites' ``spikeforge~=`` pins, so each
pinned satellite must ship its own bump in the same matrix row. Split
out of ``check_packaging_guards.py`` (#149).
"""

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

try:  # imported as part of the scripts package
    from scripts.packaging_layout import (
        COMPATIBILITY,
        CORE_DISTRIBUTION,
        PINNED_DISTRIBUTIONS,
    )
except ImportError:  # run directly as scripts/*.py
    from packaging_layout import (
        COMPATIBILITY,
        CORE_DISTRIBUTION,
        PINNED_DISTRIBUTIONS,
    )


def _releases() -> List[Dict[str, object]]:
    """Return every release entry from ``compatibility.json``."""
    with COMPATIBILITY.open("rb") as handle:
        data = json.load(handle)
    releases = data.get("releases", [])
    if not isinstance(releases, list):
        raise SystemExit("compatibility.json releases is not a list")
    return releases


def _latest_release() -> Dict[str, object]:
    """Return the newest release entry from ``compatibility.json``."""
    releases = _releases()
    if not releases:
        raise SystemExit("compatibility.json has no release entries")
    return releases[-1]


def _minor(version: object) -> Optional[Tuple[int, int]]:
    """Return the (major, minor) pair of a dotted version string."""
    match = re.match(r"^(\d+)\.(\d+)", str(version))
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)))


def _stale_satellites(
    previous: Dict[str, object],
    current: Dict[str, object],
) -> List[str]:
    """Return pinned satellites whose version did not move this row."""
    return sorted(
        distribution
        for distribution in PINNED_DISTRIBUTIONS
        if current.get(distribution) == previous.get(distribution)
    )


def check_coordinated_minor_bump(
    releases: Sequence[Dict[str, object]],
) -> Optional[str]:
    """Refuse a core MINOR bump that ships stale satellites.

    A core MINOR bump breaks the satellites' ``spikeforge~=`` pins, so
    each pinned satellite must ship its own bump in the same row.
    """
    if len(releases) < 2:
        return None
    old = _minor(releases[-2].get(CORE_DISTRIBUTION))
    new = _minor(releases[-1].get(CORE_DISTRIBUTION))
    if old is None or new is None:
        return "compatibility.json core version is not a dotted X.Y"
    if old == new:
        return None
    stale = _stale_satellites(releases[-2], releases[-1])
    if not stale:
        return None
    return f"core minor bump without coordinated bumps: {', '.join(stale)}"
