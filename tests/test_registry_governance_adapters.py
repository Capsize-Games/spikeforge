"""Registry adapters and CLI: catalog, checkpoint, and commands."""

from pathlib import Path

import pytest
import torch
from _registry_helpers import _KEY, _entry

from spikeforge.network import model_store
from spikeforge_hub import registry as reg
from spikeforge_hub.registry_errors import RegistryError


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
