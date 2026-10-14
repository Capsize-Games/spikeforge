"""Fail-closed torch loading: untrusted pickles never execute (#99)."""

from pathlib import Path
from typing import Any

import pytest
import torch

from spikeforge.cli import extract_cli
from spikeforge.network import model_store
from spikeforge.nir_bridge.errors import ExtractionError
from spikeforge_hub import artifact_readers
from spikeforge_hub.artifact_report import STATE_DICT
from spikeforge_hub.errors import HubArtifactError


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint writes into a per-test temporary directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _detonate(marker: str) -> None:
    """Write ``marker``; the stand-in for an attacker's payload."""
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write("executed")


class _Evil:
    """A checkpoint-shaped object whose unpickle executes code."""

    def __init__(self, marker: str) -> None:
        """Record where the payload would leave its mark."""
        self.marker = marker

    def __reduce__(self) -> Any:
        """Return the payload a hostile checkpoint would carry."""
        return (_detonate, (self.marker,))


def _evil_file(tmp_path: Path, name: str) -> Path:
    """Save a detonating pickle as ``name`` and return its path."""
    path = tmp_path / name
    marker = tmp_path / (path.stem + ".marker")
    torch.save(_Evil(str(marker)), str(path))
    return path


def _marker_for(path: Path) -> Path:
    """Return the marker file ``path``'s payload would write."""
    return path.parent / (path.stem + ".marker")


def test_fixture_detonates_under_raw_load(tmp_path: Path) -> None:
    """The evil fixture really executes under an unguarded load."""
    path = _evil_file(tmp_path, "evil.pt")
    torch.load(str(path), map_location="cpu", weights_only=False)
    assert _marker_for(path).exists()


def test_load_refuses_evil_checkpoint(tmp_path: Path) -> None:
    """``load`` raises a typed error and runs no payload code."""
    path = _evil_file(tmp_path, "evil.pt")
    with pytest.raises(model_store.CheckpointError):
        model_store.load("evil")
    assert not _marker_for(path).exists()


def test_list_models_reports_evil_checkpoint(tmp_path: Path) -> None:
    """A listing skips a hostile file, reports it, and runs nothing."""
    evil = _evil_file(tmp_path, "evil.pt")
    model_store.save("fine", torch.nn.Linear(4, 3), {"topology": "fc"})
    listed = {item["name"]: item for item in model_store.list_models()}
    assert listed["evil"]["unreadable"] is True
    assert "refused" in listed["evil"]["error"]
    assert "unreadable" not in listed["fine"]
    assert not _marker_for(evil).exists()


def test_hub_readers_refuse_evil_torch(tmp_path: Path) -> None:
    """Hub inspect and state-dict reads refuse hostile torch bytes."""
    path = _evil_file(tmp_path, "evil.pt")
    with pytest.raises(HubArtifactError):
        artifact_readers.inspect_torch(str(path))
    with pytest.raises(HubArtifactError):
        artifact_readers.read_state_dict(str(path))
    assert not _marker_for(path).exists()


def test_extract_load_module_refuses_evil(tmp_path: Path) -> None:
    """``extract`` refuses a module file carrying a payload."""
    path = _evil_file(tmp_path, "evil.pt")
    with pytest.raises(ExtractionError):
        extract_cli.load_module(str(path))
    assert not _marker_for(path).exists()


def test_legit_checkpoint_round_trip() -> None:
    """A saved checkpoint still loads and lists after the fix."""
    module = torch.nn.Linear(4, 3)
    model_store.save("round_trip", module, {"topology": "fc"})
    assert model_store.load("round_trip")["meta"] == {"topology": "fc"}
    listed = {item["name"]: item for item in model_store.list_models()}
    assert "unreadable" not in listed["round_trip"]


def test_legit_state_dict_still_inspects(tmp_path: Path) -> None:
    """A bare state dict still inspects as one after the fix."""
    path = tmp_path / "weights.pt"
    torch.save(torch.nn.Linear(4, 3).state_dict(), str(path))
    report = artifact_readers.inspect_torch(str(path))
    assert report.kind == STATE_DICT
    keys = {item["key"]: item["shape"] for item in report.keys}
    assert keys["weight"] == [3, 4]


def test_legit_module_still_loads_for_extract(tmp_path: Path) -> None:
    """A plain torch module still loads for ``extract`` after the fix."""
    module = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
    path = tmp_path / "module.pt"
    torch.save(module, str(path))
    loaded = extract_cli.load_module(str(path))
    assert isinstance(loaded, torch.nn.Sequential)
    for key, value in module.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)
