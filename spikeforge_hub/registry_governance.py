"""Governance views over the curated model catalog.

Split out of :mod:`spikeforge_hub.registry` so the governance layer stays
within the file-length limit. These helpers adapt catalog entries to the
governed registry shape without re-implementing the catalog loader.
"""

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from spikeforge_hub.entry import UNVERIFIED_CANDIDATE, HubEntry
from spikeforge_hub.errors import HubCatalogError
from spikeforge_hub.registry_entry import RegistryEntry, lineage_report


def _unverified_issues(entries: List[HubEntry]) -> List[str]:
    """Return one governance issue per unverified-candidate entry."""
    return [
        f"{entry.id}: unverified candidate; lifecycle maps to "
        "'deprecated' and it may not be promoted to 'prod'"
        for entry in entries
        if entry.license == UNVERIFIED_CANDIDATE
    ]


def validate_catalog(
    path: Optional[str] = None,
) -> Tuple[List[HubEntry], List[str]]:
    """Validate the catalog at ``path`` and adapt it to governance.

    Returns the valid catalog entries plus every problem, so a governance
    command can report catalog schema/version issues and lifecycle status in
    one pass without re-implementing the catalog loader.
    """
    # ``spikeforge_hub.catalog`` is shadowed at package level by the
    # ``catalog()`` function the public API exports, so the submodule must be
    # resolved through the import system rather than attribute access.
    hub_catalog = import_module("spikeforge_hub.catalog")

    target = Path(path) if path else hub_catalog.CATALOG_PATH
    try:
        entries, issues = hub_catalog.load_catalog(target)
    except HubCatalogError as error:
        return [], [str(error)]
    return entries, issues + _unverified_issues(entries)


def catalog_governance() -> List[Dict[str, Any]]:
    """Return the governed view of the bundled catalog, deterministically."""
    hub_catalog = import_module("spikeforge_hub.catalog")

    governed = [
        RegistryEntry.from_catalog_entry(entry)
        for entry in hub_catalog.entries()
    ]
    return [
        {
            **item.to_dict(),
            "lineage": lineage_report(item),
        }
        for item in sorted(governed, key=lambda item: item.id)
    ]
