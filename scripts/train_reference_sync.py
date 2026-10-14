"""Sync trained records into the hub catalog and publish weights."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from train_reference_table import REFERENCES, REPO_ROOT, Reference

#: Where a published checkpoint lands, and the catalog that points at it.
WEIGHTS_DIR = REPO_ROOT / "spikeforge_hub" / "weights"


CATALOG_PATH = REPO_ROOT / "spikeforge_hub" / "models.json"


#: Prefix for the catalog ids these checkpoints are published under.
CATALOG_PREFIX = "reference/"


def _catalog_identity(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the identity/labelling fields of a catalog entry."""
    return {
        "id": f"{CATALOG_PREFIX}{record['name']}",
        "name": (
            f"{record['dataset']} / {record['topology']} "
            f"({record['test_accuracy']}%)"
        ),
        # The checkpoint is a torch state dict for an snnTorch-backed LIF
        # network, so that is what it is labelled, not the NIR it can export.
        "framework": "snntorch",
        "kind": "state_dict",
        "source": "reference",
        # This project's own licence, for the weights. The training data's
        # own terms are a separate question and a separate field.
        "license": "BSD-3-Clause",
        "topology": record["topology"],
        "weights": f"{record['name']}.pt",
        "dataset": record["dataset"],
    }


def _catalog_scoring(
    record: Dict[str, Any], provenance: Any
) -> Dict[str, Any]:
    """Return the provenance/scoring fields of a catalog entry."""
    return {
        "dataset_license": provenance.license,
        "dataset_attribution": provenance.attribution,
        "test_accuracy": record["test_accuracy"],
        "test_samples": record["test_samples"],
        "sha256": record["sha256"],
        "size_bytes": record["checkpoint_bytes"],
    }


def _catalog_fields(
    record: Dict[str, Any], provenance: Any
) -> Dict[str, Any]:
    """Return the catalog entry's fields other than its ``notes``."""
    fields = _catalog_identity(record)
    fields.update(_catalog_scoring(record, provenance))
    return fields


def _catalog_notes(record: Dict[str, Any], reference: Reference) -> str:
    """Return the human-readable reproduction note for a catalog entry."""
    return (
        f"Trained by this project on the full {record['dataset']} "
        f"training split: {record['epochs']} epochs, "
        f"{record['num_steps']} time steps, seed {record['seed']}, CPU. "
        f"Scores {record['test_accuracy']}% on all "
        f"{record['test_samples']} held-out samples. A reference "
        f"configuration with stock hyperparameters, not a tuned attempt "
        f"at state of the art. Reproduce with: {record['command']}. "
        f"{reference.note}"
    )


def _catalog_entry(record: Dict[str, Any]) -> Dict[str, Any]:
    """Render one result record as a ``source: "reference"`` catalog entry.

    The dataset's licence and attribution are read from the provenance table
    that sits beside the dataset registry, never restated here: two copies of
    a licence is one copy that can go stale.
    """
    # Imported here, not at module scope: `spikeforge/__init__` pulls in
    # torch, and `--list` has to work without it.
    from spikeforge.data.dataset_provenance import dataset_provenance

    reference = next(
        item for item in REFERENCES if item.name == record["name"]
    )
    provenance = dataset_provenance(record["dataset"])
    entry = _catalog_fields(record, provenance)
    entry["notes"] = _catalog_notes(record, reference)
    return entry


def _rebuild_with_provenance(
    entry: Dict[str, Any], wanted: Dict[str, str]
) -> Dict[str, Any]:
    """Return ``entry`` with ``wanted`` fields placed beside ``dataset``.

    Rebuilt in order so the two fields sit beside the dataset they describe
    rather than at the end of the record.
    """
    rebuilt: Dict[str, Any] = {}
    for key, value in entry.items():
        rebuilt[key] = value
        if key == "dataset":
            rebuilt.update(wanted)
    if "dataset_license" not in rebuilt:
        rebuilt.update(wanted)
    return rebuilt


def _print_synced_entry(entry: Dict[str, Any], provenance: Any) -> None:
    """Print the provenance change for one synced catalog entry."""
    print(
        f"provenance {entry['id']}: {provenance.license} "
        f"({provenance.source})"
    )


def _sync_entry(entry: Dict[str, Any]) -> bool:
    """Refresh one catalog entry's provenance; report if it changed."""
    from spikeforge.data.dataset_provenance import dataset_provenance

    if entry.get("source") != "reference":
        return False
    provenance = dataset_provenance(str(entry["dataset"]))
    wanted = {
        "dataset_license": provenance.license,
        "dataset_attribution": provenance.attribution,
    }
    if all(entry.get(k) == v for k, v in wanted.items()):
        return False
    rebuilt = _rebuild_with_provenance(entry, wanted)
    entry.clear()
    entry.update(rebuilt)
    _print_synced_entry(entry, provenance)
    return True


def _sync_provenance() -> int:
    """Refresh every reference entry's dataset licence and attribution.

    These two fields are looked up from the provenance table by dataset name,
    not derived from the checkpoint, so they can be corrected without
    retraining — and they must be, because a licence gets verified (or
    changes upstream) long after the bytes were published. Everything that
    *does* describe the bytes (``sha256``, ``size_bytes``, ``test_accuracy``,
    ``test_samples``) is left untouched: republishing a checkpoint to fix a
    citation would replace the artifact the numbers were measured on.

    Returns the number of entries whose provenance changed.
    """
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    changed = sum(_sync_entry(entry) for entry in catalog["entries"])
    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    print(f"catalog provenance synced: {changed} entries updated")
    return changed


def _copy_checkpoints(records: List[Dict[str, Any]]) -> None:
    """Copy every trained checkpoint into the hub's weights directory."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for record in records:
        source = REPO_ROOT / record["checkpoint"]
        shutil.copy2(source, WEIGHTS_DIR / f"{record['name']}.pt")
        print(f"published {record['name']}.pt")


def _rewrite_catalog(records: List[Dict[str, Any]]) -> None:
    """Replace published reference entries with freshly built ones."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    published = {record["name"] for record in records}
    kept = [
        entry
        for entry in catalog["entries"]
        if entry.get("source") != "reference"
        or entry["id"][len(CATALOG_PREFIX):] not in published
    ]
    catalog["entries"] = [_catalog_entry(r) for r in records] + kept
    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"catalog updated: {len(records)} trained entries, "
        f"{len(catalog['entries'])} total"
    )


def _publish(records: List[Dict[str, Any]], out: Path) -> None:
    """Copy the checkpoints into the hub package and update its catalog.

    Rewriting the catalog from the results is what keeps the pinned checksum,
    size, and accuracy true of the bytes that actually shipped -- three fields
    that are worse than useless when they drift.
    """
    _copy_checkpoints(records)
    _rewrite_catalog(records)

