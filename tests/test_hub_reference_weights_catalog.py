"""The shipped reference catalog: pins, accuracy, funnel, licences."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from _reference_helpers import _WEIGHTS, _reference, _shipped

from spikeforge_hub.catalog import CATALOG_PATH, availability
from spikeforge_hub.entry import UNVERIFIED_CANDIDATE, HubEntry
from spikeforge_hub.errors import HubCatalogError
from spikeforge_hub.inspect import reference_path

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
        # Stock checkpoints reproduce through the in-repo script; the RSI
        # policy reproduces through its public sibling repo's canonical
        # command, named verbatim so the claim stays checkable.
        assert ("train_reference_models.py" in entry.notes
                or "rsi-programmer/scripts/train.py" in entry.notes), (
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


@pytest.mark.network
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


def test_every_shipped_entry_declares_its_training_data_terms() -> None:
    """No shipped checkpoint leaves what its training data permits unsaid.

    The `license` beside it covers the weights, which are this project's own.
    An entry recording one and not the other invites a reader to assume the
    first answers both questions.
    """
    from spikeforge.data.dataset_provenance import dataset_provenance

    shipped = _shipped()
    assert shipped
    for entry in shipped:
        provenance = dataset_provenance(str(entry.dataset))
        assert entry.dataset_license == provenance.license, entry.id
        assert entry.dataset_attribution == provenance.attribution, entry.id
        # Stock checkpoints are BSD-3-Clause. The RSI policy is the
        # admitted exception: Capsize-owned, trained in the public
        # rsi-programmer repo, proprietary-licensed with its manifest.
        assert entry.license in ("BSD-3-Clause",
                                 "LicenseRef-Capsize-Proprietary"), entry.id


def test_a_dataset_licence_is_never_free_text() -> None:
    """`dataset_license` is held to the same rule as `license`."""
    for value in ("see upstream", "unknown", "TBD", "CC BY-SA 4.0 probably"):
        with pytest.raises(HubCatalogError) as error:
            HubEntry.from_dict(_reference(dataset_license=value))
        assert "dataset_license" in str(error.value)


def test_an_unverified_dataset_licence_is_allowed_and_disclosed() -> None:
    """The marker records an unconfirmed licence rather than inventing one.

    It is accepted on a reference entry deliberately: four shipped MNIST
    checkpoints carry it, because MNIST's own terms could not be read from a
    primary source. Refusing it would force a guess.
    """
    entry = HubEntry.from_dict(
        _reference(dataset_license=UNVERIFIED_CANDIDATE)
    )
    assert entry.dataset_license == UNVERIFIED_CANDIDATE


def test_an_unverified_dataset_licence_does_not_gate_availability() -> None:
    """What the data permits is disclosure, not a claim about the weights."""
    entry = HubEntry.from_dict(
        _reference(dataset_license=UNVERIFIED_CANDIDATE)
    )
    available, reason = availability(entry)
    assert available is True
    assert reason is None


def test_the_catalog_round_trips_the_provenance_fields() -> None:
    """Both fields survive the schema projection in declared order."""
    entry = HubEntry.from_dict(_reference())
    payload = entry.to_dict()
    keys = list(payload)
    assert keys.index("dataset") < keys.index("dataset_license")
    assert keys.index("dataset_license") < keys.index("dataset_attribution")
    assert payload["dataset_attribution"] == (
        "Example dataset, by someone else."
    )
