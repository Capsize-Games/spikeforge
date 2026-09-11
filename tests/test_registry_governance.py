"""PT-W7 acceptance: registry governance (stages/approvals/signing/lineage).

The registry is exercised as the issue's acceptance bar states it: a bundle is
promoted with a *recorded approver* and a *verified signature*. Every refusal
path (malformed entry, bad version, terminal lifecycle, missing approver,
skipped stage, incompatible verdict, tampered signature, bad checksum) is a
named typed error, and ordering is deterministic.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from spikeforge.network import model_store
from spikeforge_hub import registry as reg
from spikeforge_hub.registry_errors import (
    RegistryApprovalError,
    RegistryCompatibilityError,
    RegistryError,
    RegistryIntegrityError,
    RegistryLifecycleError,
    RegistryPromotionError,
    RegistrySchemaError,
    RegistrySignatureError,
    RegistryVersionError,
)

_KEY = "test-registry-key"


def _entry(**overrides: Any) -> reg.RegistryEntry:
    """Return a valid dev/staging entry with optional overrides."""
    data: Dict[str, Any] = {
        "id": "streaming/anomaly",
        "name": "Streaming anomaly detector",
        "stage": "dev",
        "status": "active",
        "version": "0.1.0",
        "lineage": {
            "dataset": "uc1-telemetry",
            "model": "uc1_anomaly",
            "bundle": "uc1.spkf",
        },
    }
    data.update(overrides)
    return reg.RegistryEntry.from_dict(data)


def _signed(entry: reg.RegistryEntry) -> reg.RegistryEntry:
    """Return ``entry`` carrying a valid signature under ``_KEY``."""
    return reg.RegistryEntry(
        **{**entry.to_dict(), "signature": reg.sign_entry(entry, _KEY)}
    )


def test_good_entry_validates() -> None:
    """A well-formed entry loads and re-validates."""
    entry = _entry()
    assert entry.validated().id == "streaming/anomaly"


def test_missing_required_field_is_refused() -> None:
    """A blank required field is a typed schema error."""
    with pytest.raises(RegistrySchemaError):
        reg.RegistryEntry.from_dict({"id": "m/x", "stage": "dev"})


def test_unknown_stage_is_refused() -> None:
    """An unknown stage is refused by name."""
    with pytest.raises(RegistrySchemaError):
        _entry(stage="qa")


def test_unknown_status_is_refused() -> None:
    """An unknown lifecycle status is refused by name."""
    with pytest.raises(RegistrySchemaError):
        _entry(status="frozen")


def test_bad_version_is_refused() -> None:
    """A non-numeric version is a typed version error."""
    with pytest.raises(RegistryVersionError):
        _entry(version="latest")


def test_document_schema_version_mismatch_is_refused() -> None:
    """A registry document pinned to another schema is refused."""
    with pytest.raises(RegistryVersionError):
        reg.Registry.from_dict({"version": 99, "entries": []})


def test_promotion_requires_an_approver() -> None:
    """A promotion with no approver is refused as an approval error."""
    with pytest.raises(RegistryApprovalError):
        reg.promote(_entry(), "staging", "  ", _KEY)


def test_promotion_cannot_skip_a_stage() -> None:
    """``dev -> prod`` is refused; only the next stage is allowed."""
    with pytest.raises(RegistryPromotionError):
        reg.promote(_entry(compat=reg.EXACT), "prod", "alice", _KEY)


def test_deprecated_entry_cannot_be_promoted() -> None:
    """A deprecated entry is refused by lifecycle status."""
    with pytest.raises(RegistryLifecycleError):
        reg.promote(_entry(status="deprecated"), "staging", "alice", _KEY)


def test_prod_requires_a_compatible_verdict() -> None:
    """A ``prod`` promotion needs an exact/mappable verdict."""
    staged = _signed(reg.promote(_entry(), "staging", "alice", _KEY))
    with pytest.raises(RegistryCompatibilityError):
        reg.promote(staged, "prod", "bob", _KEY)


def test_prod_promotion_records_approver_and_signature() -> None:
    """The acceptance path: verify the signature, then promote to prod."""
    dev = _signed(_entry(compat=reg.EXACT))
    assert reg.verify_entry(dev, _KEY) is True
    staged = reg.promote(dev, "staging", "alice", _KEY, at=1.0)
    assert staged.stage == "staging"
    assert staged.approver == "alice"
    assert reg.verify_entry(staged, _KEY) is True
    prod = reg.promote(staged, "prod", "bob", _KEY, at=2.0)
    assert prod.stage == "prod"
    assert prod.approver == "bob"
    assert prod.approved_at == 2.0
    assert reg.verify_entry(prod, _KEY) is True


def test_tampered_signature_is_refused() -> None:
    """A wrong signature is refused with the expected digest."""
    entry = _entry(signature="deadbeef")
    with pytest.raises(RegistrySignatureError):
        reg.verify_entry(entry, _KEY)


def test_artifact_checksum_mismatch_is_refused(tmp_path: Path) -> None:
    """An artifact whose bytes do not match its recorded digest is refused."""
    artifact = tmp_path / "bundle.bin"
    artifact.write_bytes(b"not the recorded bytes")
    entry = _entry(
        artifact_path=str(artifact),
        artifact_sha256="0" * 64,
    )
    with pytest.raises(RegistryIntegrityError):
        reg.verify_artifact(entry)
    with pytest.raises(RegistryIntegrityError):
        reg.promote(entry, "staging", "alice", _KEY)


def test_artifact_checksum_match_reports_verified(tmp_path: Path) -> None:
    """A matching recorded digest reports ``verified``."""
    from spikeforge_hub.verify import file_sha256

    artifact = tmp_path / "bundle.bin"
    artifact.write_bytes(b"the recorded bytes")
    entry = _entry(
        artifact_path=str(artifact),
        artifact_sha256=file_sha256(str(artifact)),
    )
    assert reg.verify_artifact(entry)["status"] == "verified"


def test_registry_ordering_is_deterministic() -> None:
    """Entries order by stage, then id, then version."""
    entries = (
        _entry(id="b/two", stage="prod"),
        _entry(id="a/one", stage="staging"),
        _entry(id="a/one", stage="dev"),
    )
    loaded = reg.Registry(entries=entries)
    ordered = loaded.ordered()
    assert [(item.stage, item.id) for item in ordered] == [
        ("dev", "a/one"),
        ("staging", "a/one"),
        ("prod", "b/two"),
    ]
    assert reg.Registry(entries=tuple(reversed(entries))).ordered() == ordered


def test_registry_load_collects_issues(tmp_path: Path) -> None:
    """A malformed entry is reported as an issue, not silently dropped."""
    document = {
        "version": reg.REGISTRY_SCHEMA_VERSION,
        "entries": [_entry().to_dict(), {"id": "broken"}],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = reg.Registry.load(str(path))
    assert len(loaded.entries) == 1
    assert len(loaded.issues) == 1
    assert "broken" in loaded.issues[0]


def test_registry_round_trips_through_disk(tmp_path: Path) -> None:
    """A saved registry reloads to the same ordered entries."""
    path = str(tmp_path / "registry.json")
    original = reg.Registry().add(_entry())
    original.save(path)
    reloaded = reg.Registry.load(path)
    assert reloaded.to_dict() == original.to_dict()


def test_lineage_report_names_missing_links() -> None:
    """The lineage chain is ordered and names the links still missing."""
    report = reg.lineage_report(_entry())
    assert [item["link"] for item in report["chain"]] == [
        "dataset",
        "model",
        "bundle",
    ]
    assert report["complete"] is False
    assert report["missing"] == ["deployment"]


def test_validate_catalog_accepts_the_bundled_catalog() -> None:
    """The shipped catalog passes governance with no schema issues."""
    entries, issues = reg.validate_catalog()
    assert entries
    assert not [issue for issue in issues if "invalid catalog entry" in issue]


def test_from_catalog_entry_maps_unverified_to_deprecated() -> None:
    """An unverified candidate adapts to the terminal lifecycle."""
    from spikeforge_hub.entry import UNVERIFIED_CANDIDATE, HubEntry

    entry = HubEntry(
        id="remote/candidate",
        name="Candidate",
        framework="nir",
        kind="nir_graph",
        source="bundled",
        license=UNVERIFIED_CANDIDATE,
        notes="candidate",
    )
    governed = reg.RegistryEntry.from_catalog_entry(entry)
    assert governed.status == "deprecated"


def test_from_checkpoint_records_the_artifact_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint-backed entry records the model store's path and digest."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))
    module = torch.nn.Linear(2, 2)
    model_store.save("governed_model", module, {"topology": "sequence_mlp"})
    entry = reg.RegistryEntry.from_checkpoint(
        "governed_model", compat=reg.EXACT
    )
    assert entry.artifact_path is not None
    assert entry.artifact_sha256 is not None
    assert reg.verify_artifact(entry)["status"] == "verified"
    assert entry.lineage["model"] == "governed_model"


def test_registry_cli_list_validate_promote_verify(tmp_path: Path) -> None:
    """The CLI lists, validates, promotes, and verifies end to end."""
    from spikeforge_hub import registry_cli

    path = str(tmp_path / "registry.json")
    reg.Registry().add(_entry(compat=reg.EXACT)).save(path)

    assert registry_cli.main(["list", "--registry", path]) == 0
    assert registry_cli.main(["validate", "--registry", path]) == 0
    assert (
        registry_cli.main(
            [
                "promote",
                "streaming/anomaly",
                "--to",
                "staging",
                "--approver",
                "alice",
                "--registry",
                path,
                "--key",
                _KEY,
            ]
        )
        == 0
    )
    assert (
        registry_cli.main(
            ["verify", "streaming/anomaly", "--registry", path, "--key", _KEY]
        )
        == 0
    )
    promoted = reg.Registry.load(path).get("streaming/anomaly")
    assert promoted is not None
    assert promoted.stage == "staging"
    assert promoted.approver == "alice"


def test_registry_cli_refuses_missing_approver(tmp_path: Path) -> None:
    """A CLI promotion without an approver fails non-zero, no traceback."""
    from spikeforge_hub import registry_cli

    path = str(tmp_path / "registry.json")
    reg.Registry().add(_entry(compat=reg.EXACT)).save(path)
    assert (
        registry_cli.main(
            [
                "promote",
                "streaming/anomaly",
                "--to",
                "staging",
                "--approver",
                " ",
                "--registry",
                path,
                "--key",
                _KEY,
            ]
        )
        == 1
    )
    assert isinstance(RegistryError("x"), RegistryError)
