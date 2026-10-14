"""Reference-entry builders shared by the hub weight tests."""

from pathlib import Path
from typing import Any, Dict

from spikeforge_hub.catalog import entries
from spikeforge_hub.entry import WEIGHTS_DIR

_PACKAGE = Path(__file__).resolve().parent.parent / "spikeforge_hub"
_WEIGHTS = _PACKAGE / WEIGHTS_DIR

def _reference(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal schema-valid reference entry, with overrides."""
    data: Dict[str, Any] = {
        "id": "reference/example",
        "name": "Example (94.0%)",
        "framework": "snntorch",
        "kind": "state_dict",
        "source": "reference",
        "license": "BSD-3-Clause",
        "notes": "Trained by this project.",
        "weights": "example.pt",
        "dataset": "mnist",
        "dataset_license": "CC-BY-SA-4.0",
        "dataset_attribution": "Example dataset, by someone else.",
        "sha256": "0" * 64,
        "test_accuracy": 94.0,
        "test_samples": 10000,
    }
    data.update(overrides)
    return data


def _shipped() -> list:
    """Return every catalog entry that carries trained weights."""
    return [entry for entry in entries() if entry.trained]
