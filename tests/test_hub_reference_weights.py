"""Trained reference entries: schema and weight resolution."""

from pathlib import Path
from typing import Any

import pytest
import torch
from _reference_helpers import _WEIGHTS, _reference, _shipped

from spikeforge_hub.catalog import availability
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubArtifactError, HubCatalogError
from spikeforge_hub.inspect import (
    STATE_DICT,
    inspect_artifact,
    reference_path,
    resolve_path,
)

# --- schema ---------------------------------------------------------------


def test_a_reference_entry_validates() -> None:
    """The generated shape is the shape validation accepts."""
    entry = HubEntry.from_dict(_reference())
    assert entry.trained is True
    assert entry.weights == "example.pt"
    assert entry.test_accuracy == 94.0


def test_a_bundled_entry_is_not_trained() -> None:
    """``trained`` separates the two products the catalog now holds."""
    entry = HubEntry.from_dict(
        {
            "id": "nir/example",
            "name": "Example",
            "framework": "nir",
            "kind": "nir_graph",
            "source": "bundled",
            "license": "BSD-3-Clause",
            "notes": "Rendered from a preset.",
            "topology": "fc_small",
        }
    )
    assert entry.trained is False


@pytest.mark.parametrize(
    ("field", "detail"),
    [
        ("weights", "weights"),
        ("sha256", "sha256"),
        ("dataset", "dataset"),
        ("dataset_license", "dataset_license"),
        ("dataset_attribution", "dataset_attribution"),
        ("test_accuracy", "test_accuracy"),
    ],
)
def test_a_reference_entry_must_declare_its_evidence(
    field: str, detail: str
) -> None:
    """Dropping any load-bearing field is rejected by name."""
    data = _reference()
    data[field] = None
    with pytest.raises(HubCatalogError) as error:
        HubEntry.from_dict(data)
    assert detail in str(error.value)


@pytest.mark.parametrize(
    "weights",
    ["../escape.pt", "nested/file.pt", "notapt.bin", "/absolute.pt"],
)
def test_weights_must_be_a_bare_filename(weights: str) -> None:
    """A weights name is a filename, never a path into the filesystem."""
    with pytest.raises(HubCatalogError) as error:
        HubEntry.from_dict(_reference(weights=weights))
    assert "weights" in str(error.value)


@pytest.mark.parametrize("accuracy", [-1, 100.1, "94", True])
def test_an_impossible_accuracy_is_rejected(accuracy: Any) -> None:
    """A percentage outside 0-100, or a non-number, fails validation."""
    with pytest.raises(HubCatalogError):
        HubEntry.from_dict(_reference(test_accuracy=accuracy))


# --- resolution -----------------------------------------------------------


def test_reference_weights_are_loaded_not_rebuilt(tmp_path: Path) -> None:
    """The whole point: the trained bytes are returned, not a fresh preset."""
    entry = _shipped()
    if not entry:
        pytest.skip("no reference entries are shipped in this catalog")
    resolved = Path(resolve_path(entry[0]))
    assert resolved.parent == _WEIGHTS
    assert resolved.name == entry[0].weights
    loaded = torch.load(resolved, map_location="cpu", weights_only=False)
    assert loaded["state_dict"], "checkpoint carries no weights"


def test_a_checksum_mismatch_is_reported_not_loaded() -> None:
    """A trained entry whose bytes changed must fail loudly."""
    shipped = _shipped()
    if not shipped:
        pytest.skip("no reference entries are shipped in this catalog")
    tampered = HubEntry.from_dict(
        {**shipped[0].to_dict(), "sha256": "f" * 64}
    )
    with pytest.raises(HubArtifactError) as error:
        reference_path(tampered)
    assert "checksum" in str(error.value)


def test_missing_packaged_weights_name_the_install() -> None:
    """An incomplete install is reported as such, not as a mystery."""
    entry = HubEntry.from_dict(_reference(weights="not-shipped.pt"))
    with pytest.raises(HubArtifactError) as error:
        reference_path(entry)
    assert "missing from this install" in str(error.value)


def test_a_reference_entry_inspects_as_a_state_dict() -> None:
    """The import funnel's first gate classifies the checkpoint correctly."""
    shipped = _shipped()
    if not shipped:
        pytest.skip("no reference entries are shipped in this catalog")
    report = inspect_artifact(resolve_path(shipped[0]))
    assert report.kind == STATE_DICT
    assert report.keys, "no tensor keys were reported"


def test_a_reference_entry_is_available_without_any_extra() -> None:
    """Shipped weights need no network and no optional dependency."""
    shipped = _shipped()
    if not shipped:
        pytest.skip("no reference entries are shipped in this catalog")
    available, reason = availability(shipped[0])
    assert available is True
    assert reason is None
