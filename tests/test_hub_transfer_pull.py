"""Pull orchestration and the ``.spkf`` import funnel."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from _transfer_helpers import _save_checkpoint

from spikeforge.network import model_store
from spikeforge.serving import bundle as bundle_mod
from spikeforge_hub import cache, download_cli, transfer
from spikeforge_hub.artifact_readers import inspect_file
from spikeforge_hub.errors import HubDownloadError
from spikeforge_hub.import_model import import_model

pytest.importorskip("nir")

_ENTRY: Dict[str, Any] = {
    "id": "@ada/mnist-fc",
    "name": "mnist-fc",
    "framework": "spikeforge",
    "kind": "bundle",
    "source": "community",
    "license": "BSD-3-Clause",
    "notes": "Community upload, version 1. Checks: unchecked.",
    "url": "https://cdn.test/mnist-fc.spkf",
}

@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the hub cache and model store into a temporary directory."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path / "hub"))
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path / "models"))
    return tmp_path


def _spkf_bytes(
    name: str, dest: Path, metrics: Optional[Dict[str, Any]] = None
) -> bytes:
    """Build saved model ``name`` and return its ``.spkf`` bytes."""
    out = str(dest / f"{name}.spkf")
    bundle_mod.build(name, out=out, expected_metrics=metrics or {})
    return Path(out).read_bytes()


def _seed(root: Path, entries: List[Dict[str, Any]]) -> None:
    """Write ``entries`` as the cached hosted index under ``root``."""
    dest = root / "hub" / ".community"
    dest.mkdir(parents=True)
    payload = {"version": 1, "entries": entries}
    (dest / "index.json").write_text(json.dumps(payload), encoding="utf-8")


def _fake_index(
    monkeypatch: pytest.MonkeyPatch, refreshed: bool = True
) -> None:
    """Pretend the hosted index refreshed (or went stale)."""
    fetched = SimpleNamespace(entries=[], issues=[], refreshed=refreshed)
    monkeypatch.setattr(transfer, "fetch_index", lambda: fetched)


def _fake_fetch(
    monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> List[str]:
    """Serve ``payload`` for every artifact fetch, recording URLs."""
    seen: List[str] = []

    def fake_fetch(url: str, path: str, **_kwargs: Any) -> str:
        seen.append(url)
        Path(path).write_bytes(payload)
        return path

    monkeypatch.setattr(download_cli, "fetch_url", fake_fetch)
    monkeypatch.setattr(transfer, "fetch_url", fake_fetch)
    return seen


def test_parse_pull_id_splits_and_refuses() -> None:
    """Both spellings split; anything else is refused by name."""
    plain = transfer.parse_pull_id("@ada/mnist-fc")
    assert (plain.id, plain.handle, plain.slug, plain.version) == (
        "@ada/mnist-fc", "ada", "mnist-fc", None,
    )
    pinned = transfer.parse_pull_id("  @ada/mnist-fc@2  ")
    assert (pinned.id, pinned.version) == ("@ada/mnist-fc", "2")
    for bad in ("", "nir/fc_small", "@ada", "@/x", "@a/", "@a/b@",
                "@a/b@v@w", "@a/b/c", "@a/b@v/x"):
        with pytest.raises(HubDownloadError, match="pull id"):
            transfer.parse_pull_id(bad)


def test_bundle_inspects_as_a_state_dict(sandbox: Path) -> None:
    """A ``.spkf`` reports its weight keys like the checkpoint it came from."""
    _save_checkpoint()
    blob = _spkf_bytes("transfer_ckpt", sandbox)
    path = sandbox / "probe.spkf"
    path.write_bytes(blob)
    report = inspect_file(str(path))
    assert report.kind == "state_dict"
    assert {key["key"] for key in report.keys} >= {
        "fc1.weight", "fc2.weight",
    }
    assert report.notes[0].startswith("spkf bundle")


def test_bundle_import_matches_checkpoint_import(sandbox: Path) -> None:
    """The funnel judges a bundle exactly as it judges its checkpoint."""
    _save_checkpoint()
    blob = _spkf_bytes("transfer_ckpt", sandbox)
    path = sandbox / "parity.spkf"
    path.write_bytes(blob)
    from_bundle = import_model(path=str(path))
    from_ckpt = import_model(path=model_store.path_for("transfer_ckpt"))
    assert from_bundle["verdict"] == from_ckpt["verdict"]
    assert from_bundle["promoted"] == from_ckpt["promoted"]
    assert from_bundle.get("reason") == from_ckpt.get("reason")


def test_pull_downloads_and_imports_through_the_funnel(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pull fetches the index URL, then the funnel judges the bytes."""
    _save_checkpoint()
    blob = _spkf_bytes("transfer_ckpt", sandbox)
    # The hosted index pins the digest it measured on arrival, so the
    # pulled bytes verify at import; the seed mirrors that contract.
    _seed(sandbox, [{
        **_ENTRY,
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }])
    seen = _fake_fetch(monkeypatch, blob)
    _fake_index(monkeypatch)
    report = transfer.pull_model("@ada/mnist-fc")
    assert seen == ["https://cdn.test/mnist-fc.spkf"]
    assert report["downloaded"] is True
    assert report["index_refreshed"] is True
    assert report["index_issues"] == []
    assert report["import"]["verdict"]["verdict"] == "mappable"
    assert report["expected_metrics"] == {}
    assert report["metrics_claim"] is None
    again = transfer.pull_model("@ada/mnist-fc")
    assert again["downloaded"] is False
    assert again["download"] is None
    assert len(seen) == 1


def test_pull_surfaces_claimed_metrics(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pull reports the bundle's metrics as uploader-claimed, never more."""
    _save_checkpoint()
    claimed = {
        "test_accuracy": 94.29,
        "test_samples": 10000,
        "test_dataset": "mnist",
    }
    blob = _spkf_bytes("transfer_ckpt", sandbox, claimed)
    _seed(sandbox, [{
        **_ENTRY,
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }])
    _fake_fetch(monkeypatch, blob)
    _fake_index(monkeypatch)
    report = transfer.pull_model("@ada/mnist-fc")
    assert report["expected_metrics"] == claimed
    assert report["metrics_claim"] == "uploader-claimed"


def test_pull_pinned_version_downloads_its_own_bytes(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pinned pull hits the version route and imports the bytes by path."""
    _save_checkpoint()
    blob = _spkf_bytes("transfer_ckpt", sandbox)
    seen = _fake_fetch(monkeypatch, blob)
    _fake_index(monkeypatch)
    report = transfer.pull_model("@ada/mnist-fc@2")
    assert seen[0].endswith("/v1/models/ada/mnist-fc/versions/2/download")
    assert report["version"] == "2"
    assert report["download"]["verified"] is False
    fetched = Path(report["download"]["path"])
    assert fetched.parent.name == "@ada__mnist-fc"
    assert fetched.name == "mnist-fc-2.spkf"
    assert fetched.is_file()
    assert report["import"]["verdict"]["verdict"] == "mappable"


def test_pull_unknown_id_names_staleness(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unresolvable id with a stale index says the cache may lag."""
    _fake_index(monkeypatch, refreshed=False)
    with pytest.raises(HubDownloadError, match="stale"):
        transfer.pull_model("@ada/mnist-fc")


def test_pull_non_community_source_is_refused(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An at-id entry that is not community-owned is refused, not pulled."""
    _seed(sandbox, [{**_ENTRY, "source": "url", "sha256": "0" * 64}])
    _fake_index(monkeypatch)
    with pytest.raises(HubDownloadError, match="community id"):
        transfer.pull_model("@ada/mnist-fc")
