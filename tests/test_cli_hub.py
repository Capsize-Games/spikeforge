"""The ``spikeforge-hub`` CLI subcommands."""

import json
from pathlib import Path
from typing import Any

import pytest

from spikeforge.network import model_store
from spikeforge_hub import cache, cli

pytest.importorskip("nir")


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the hub cache and model store into a temporary directory."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path / "hub"))
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path / "models"))
    return tmp_path


def test_list_prints_the_catalog(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``list`` prints at least ten entries and no validation issues."""
    assert cli.main(["list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["entries"]) >= 10
    assert payload["issues"] == []


def test_list_filters_by_framework(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``list --framework`` keeps only that framework's entries."""
    assert cli.main(["list", "--framework", "nir"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["entries"]
    assert all(item["framework"] == "nir" for item in payload["entries"])


def test_search_prints_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``search`` prints the query and its bounded results."""
    assert cli.main(["search", "conv", "--limit", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "conv"
    assert len(payload["results"]) <= 3


def test_inspect_prints_structure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``inspect`` prints the artifact's structural report."""
    assert cli.main(["inspect", "nir/fc_small"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "nir_graph"
    assert payload["nodes"]


def test_inspect_unknown_entry_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unknown id exits non-zero with a named reason."""
    assert cli.main(["inspect", "nope"]) == 1
    assert "unknown catalog entry" in capsys.readouterr().out


def test_import_exact_promotes_and_exits_zero(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``import`` of a compatible artifact prints the report and exits 0."""
    assert cli.main(["import", "nir/fc_small"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["promoted"] is True
    assert payload["verdict"]["verdict"] == "exact"


def test_import_incompatible_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An incompatible verdict gates the exit code, still printing why."""

    def fake(**kwargs: Any) -> dict:
        return {"verdict": {"verdict": "incompatible", "mismatches": []}}

    monkeypatch.setattr(cli, "import_model", fake)
    assert cli.main(["import", "nir/fc_small"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"]["verdict"] == "incompatible"


def test_download_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed verification gates the exit code."""
    monkeypatch.setattr(
        cli, "hub_download", lambda entry, dest=None: {"status": "failed"}
    )
    assert cli.main(["download", "nir/fc_small"]) == 1
    capsys.readouterr()


def test_download_no_verify_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--no-verify`` reports the failure but does not gate the exit."""
    monkeypatch.setattr(
        cli, "hub_download", lambda entry, dest=None: {"status": "failed"}
    )
    assert cli.main(["download", "nir/fc_small", "--no-verify"]) == 0
    capsys.readouterr()


def test_download_success_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A verified download exits zero."""
    monkeypatch.setattr(
        cli, "hub_download", lambda entry, dest=None: {"status": "verified"}
    )
    assert cli.main(["download", "nir/fc_small"]) == 0
    capsys.readouterr()
