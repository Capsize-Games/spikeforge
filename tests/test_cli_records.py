"""CLI ``records`` subcommands: JSON output and exit codes."""

import json
from typing import Any, Dict, Optional

import pytest
import torch

from snn_interpreter.cli import verify
from snn_interpreter.network import model_store


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint writes into a per-test temporary directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _save(
    name: str,
    dataset: str,
    coding: str,
    accuracy: Optional[float],
    manifest: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one checkpoint with the given searchable metadata."""
    meta = {
        "dataset": dataset,
        "topology": "fc_legacy",
        "coding": coding,
        "input_mode": coding,
        "device": "cpu",
    }
    history = (
        [] if accuracy is None else [{"step": 1, "test_accuracy": accuracy}]
    )
    model_store.save(name, torch.nn.Linear(4, 2), meta, history, manifest)


def test_records_list_returns_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``records list`` prints filterable JSON and exits zero."""
    _save("mnist_rate", "mnist", "rate", 90.0)
    _save("cifar_rate", "cifar10", "rate", 50.0)
    assert verify.main(["records", "list", "--dataset", "mnist"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["name"] for item in payload["models"]] == ["mnist_rate"]


def test_records_list_min_accuracy_filter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The minimum-accuracy filter drops lower-scoring checkpoints."""
    _save("low", "mnist", "rate", 40.0)
    _save("high", "mnist", "rate", 80.0)
    assert verify.main(["records", "list", "--min-accuracy", "60"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["name"] for item in payload["models"]] == ["high"]


def test_records_manifest_round_trip(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``records manifest`` prints the stored manifest and exits zero."""
    _save("tracked", "mnist", "rate", 70.0, {"config_hash": "abc"})
    assert verify.main(["records", "manifest", "tracked"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["available"] is True
    assert payload["manifest"]["config_hash"] == "abc"


def test_records_manifest_missing_is_not_found(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing checkpoint exits non-zero with a plain message."""
    assert verify.main(["records", "manifest", "ghost"]) == 1
    assert "ghost" in capsys.readouterr().out


def test_records_diff_classifies_changes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``records diff`` prints the classified metadata union."""
    _save("a", "mnist", "rate", 90.0)
    _save("b", "cifar10", "rate", 50.0)
    assert verify.main(["records", "diff", "a", "b"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["changed"] == ["dataset"]


def test_records_diff_missing_is_not_found(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A diff against a missing checkpoint exits non-zero."""
    _save("a", "mnist", "rate", 90.0)
    assert verify.main(["records", "diff", "a", "ghost"]) == 1
