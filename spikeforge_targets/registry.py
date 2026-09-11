"""Lookup over the built-in target specifications.

The index is built once at import time from :func:`catalog.builtin_targets`,
which touches no SDK. Availability is resolved on demand through the isolated
:mod:`probe`, so a query is the only thing that can attempt an import.
"""

from typing import Dict, List

from spikeforge_targets import probe
from spikeforge_targets.catalog import builtin_targets
from spikeforge_targets.target_spec import TargetSpec

_TARGETS: Dict[str, TargetSpec] = {
    spec.name: spec for spec in builtin_targets()
}


def target_names() -> List[str]:
    """Return the built-in target names in registry order."""
    return list(_TARGETS)


def get_target(name: str) -> TargetSpec:
    """Return the target called ``name`` or raise ``ValueError``."""
    try:
        return _TARGETS[name]
    except KeyError:
        raise ValueError(f"unknown target: {name!r}") from None


def available(name: str) -> bool:
    """Return True when ``name``'s SDK is installed and importable."""
    return probe.extra_available(get_target(name).extra)


def available_names() -> List[str]:
    """Return the names of every currently-available target."""
    return [name for name in _TARGETS if available(name)]


def unavailable_names() -> List[str]:
    """Return the names of every target whose SDK is absent."""
    return [name for name in _TARGETS if not available(name)]
