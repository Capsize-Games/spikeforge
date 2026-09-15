"""Trained reference entries: schema, resolution, and the shipped catalog.

The hub's long-standing gap was that every entry was structure with
freshly-initialised weights. ``source: "reference"`` closes it, and these
tests hold the line the curation policy draws: an entry that claims trained
weights must name them, pin their checksum, and say what they score.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from spikeforge_hub.catalog import CATALOG_PATH, availability, entries
from spikeforge_hub.entry import WEIGHTS_DIR, HubEntry
from spikeforge_hub.errors import HubArtifactError, HubCatalogError
from spikeforge_hub.inspect import (
    STATE_DICT,
    inspect_artifact,
    reference_path,
    resolve_path,
)

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
        "sha256": "0" * 64,
        "test_accuracy": 94.0,
        "test_samples": 10000,
    }
    data.update(overrides)
    return data


def _shipped() -> list:
    """Return every catalog entry that carries trained weights."""
    return [entry for entry in entries() if entry.trained]


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
        ("test_accuracy", "test_accuracy"),
    ],
)
def test_a_reference_entry_must_declare_its_evidence(
    field: str, detail: str
) -> None:
    """Dropping any of the four load-bearing fields is rejected by name."""
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


# --- the shipped catalog --------------------------------------------------


def test_the_catalog_ships_trained_entries() -> None:
    """The hub's headline gap: it must no longer be zero."""
    assert _shipped(), (
        "the catalog carries no trained weights; regenerate them with "
        "python scripts/train_reference_models.py --publish"
    )


def test_every_shipped_checksum_matches_its_file() -> None:
    """The pinned checksum describes the bytes that actually shipped."""
    for entry in _shipped():
        path = _WEIGHTS / str(entry.weights)
        assert path.is_file(), f"{entry.id}: {entry.weights} is not shipped"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry.sha256, f"{entry.id}: checksum drifted"
        assert path.stat().st_size == entry.size_bytes, (
            f"{entry.id}: size_bytes drifted"
        )


def test_no_shipped_weight_file_is_orphaned() -> None:
    """Every packaged checkpoint is reachable through the catalog."""
    if not _WEIGHTS.is_dir():
        pytest.skip("no packaged weights directory")
    named = {str(entry.weights) for entry in _shipped()}
    on_disk = {path.name for path in _WEIGHTS.glob("*.pt")}
    assert on_disk == named, (
        "packaged weights and catalog entries disagree: "
        f"orphaned {sorted(on_disk - named)}, "
        f"missing {sorted(named - on_disk)}"
    )


def test_the_packaged_weights_stay_within_their_size_budget() -> None:
    """CURATION.md commits to a few megabytes; hold it to that."""
    if not _WEIGHTS.is_dir():
        pytest.skip("no packaged weights directory")
    total = sum(path.stat().st_size for path in _WEIGHTS.glob("*.pt"))
    budget = 8 * 1024 * 1024
    assert total <= budget, (
        f"packaged weights total {total} bytes, over the {budget}-byte "
        "budget; move large checkpoints behind source 'url'"
    )


def test_every_trained_entry_names_a_reproduction_command() -> None:
    """A published number a reader cannot reproduce is not evidence."""
    for entry in _shipped():
        assert "train_reference_models.py" in entry.notes, (
            f"{entry.id}: notes do not name the command that reproduces it"
        )


def test_the_catalog_file_and_the_loaded_entries_agree() -> None:
    """A reference entry must survive a round trip through validation."""
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    on_disk = [
        item for item in raw["entries"] if item.get("source") == "reference"
    ]
    assert len(on_disk) == len(_shipped())


def test_a_shipped_checkpoint_restores_into_its_topology() -> None:
    """The weights load into the topology the entry names, key for key."""
    shipped = _shipped()
    if not shipped:
        pytest.skip("no reference entries are shipped in this catalog")
    from spikeforge.topology.registry import build_topology

    entry = shipped[0]
    checkpoint = torch.load(
        reference_path(entry), map_location="cpu", weights_only=False
    )
    params = checkpoint["meta"]["topology_params"]
    _spec, module = build_topology(str(entry.topology), params)
    module.load_state_dict(checkpoint["state_dict"])


def test_a_restored_checkpoint_reproduces_its_published_accuracy() -> None:
    """The number in the catalog is the number the weights actually score.

    Scored on one batch rather than the full split so the suite stays fast; a
    checkpoint that is mislabelled, mis-restored, or silently untrained lands
    near chance, which this catches without re-running the published
    evaluation.
    """
    shipped = _shipped()
    if not shipped:
        pytest.skip("no reference entries are shipped in this catalog")
    entry = shipped[0]
    from spikeforge import TrainingEngine
    from spikeforge.data.data_loader import build_loader

    checkpoint = torch.load(
        reference_path(entry), map_location="cpu", weights_only=False
    )
    meta = checkpoint["meta"]
    # The model store addresses checkpoints by name inside MODEL_DIR, so a
    # packaged artifact is loaded into the built topology directly; the
    # user-facing route for the same thing is ``spikeforge-hub import``,
    # covered below.
    engine = TrainingEngine(
        dataset=str(entry.dataset),
        topology=str(entry.topology),
        hidden=int(meta["hidden"]),
        beta=float(meta["beta"]),
        num_steps=int(meta["num_steps"]),
        device="cpu",
    )
    engine.net.load_state_dict(checkpoint["state_dict"])

    loader = build_loader(str(entry.dataset), 1, 512, train=False)
    inputs, targets = next(iter(loader))
    scored = 100.0 * float(
        (engine.predict(inputs).cpu() == targets).float().mean()
    )
    published = float(entry.test_accuracy or 0.0)
    assert scored >= published - 10.0, (
        f"{entry.id}: restored checkpoint scores {scored:.1f}% on a test "
        f"batch against a published {published}%"
    )


def test_a_reference_entry_survives_the_import_funnel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The user-facing route works end to end: inspect, compat, promote.

    This is the journey the hub exists for, and until reference entries
    existed it could only ever hand back freshly-initialised weights.
    """
    pytest.importorskip("nir")
    from spikeforge import config
    from spikeforge.network import model_store
    from spikeforge_hub.import_model import import_model

    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(config, "MODEL_DIR", str(tmp_path), raising=False)

    shipped = _shipped()
    if not shipped:
        pytest.skip("no reference entries are shipped in this catalog")
    result = import_model(entry_id=shipped[0].id)
    assert result["verdict"]["verdict"] != "incompatible", result["verdict"]
