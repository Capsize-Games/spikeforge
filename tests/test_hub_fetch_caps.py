"""Per-entry fetch caps and snapshot verification on the hub path."""

import hashlib
from pathlib import Path
from typing import Any, Dict

import pytest

from spikeforge_hub import download_cli
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubCatalogError, HubDownloadError


def _sha(blob: bytes) -> str:
    """Return the hex sha256 digest of ``blob``."""
    return hashlib.sha256(blob).hexdigest()


def _entry(source: str, **overrides: Any) -> HubEntry:
    """Return a schema-valid entry for the given source."""
    data: Dict[str, Any] = {
        "id": "nir/example",
        "name": "Example",
        "framework": "nir",
        "kind": "nir_graph",
        "source": source,
        "license": "BSD-3-Clause",
        "notes": "unit-test entry",
    }
    data.update(overrides)
    return HubEntry.from_dict(data)


def _recording_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> Dict[str, Any]:
    """Serve a fixed payload for every fetch, recording kwargs."""
    seen: Dict[str, Any] = {}

    def fake_fetch(url: str, path: str, **kwargs: Any) -> str:
        seen.update(kwargs)
        Path(path).write_bytes(b"payload")
        return path

    monkeypatch.setattr(download_cli, "fetch_url", fake_fetch)
    return seen


def test_entry_cap_reaches_the_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An entry-level max_bytes narrows the worker's fetch cap."""
    seen = _recording_fetch(monkeypatch)
    blob = b"payload"
    entry = _entry(
        "url",
        url="https://example.test/m",
        sha256=_sha(blob),
        size_bytes=len(blob),
        max_bytes=1234,
    )
    download_cli.download_entry(entry, str(tmp_path))
    assert seen["max_bytes"] == 1234


def test_missing_entry_cap_uses_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An entry without max_bytes fetches under the 2 GiB default."""
    seen = _recording_fetch(monkeypatch)
    blob = b"payload"
    entry = _entry(
        "url",
        url="https://example.test/m",
        sha256=_sha(blob),
        size_bytes=len(blob),
    )
    download_cli.download_entry(entry, str(tmp_path))
    assert seen["max_bytes"] == download_cli.DEFAULT_MAX_BYTES


@pytest.mark.parametrize("limit", [0, -5, "100", True, 1.5])
def test_max_bytes_schema_rejects_nonpositive(limit: Any) -> None:
    """A non-positive-integer max_bytes fails catalog validation."""
    with pytest.raises(HubCatalogError):
        _entry("bundled", max_bytes=limit)


def test_max_bytes_schema_round_trips() -> None:
    """A positive max_bytes loads and survives to_dict."""
    entry = _entry("bundled", max_bytes=1024)
    assert entry.max_bytes == 1024
    assert entry.to_dict()["max_bytes"] == 1024


def test_snapshot_over_cap_deleted_and_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An over-cap snapshot is deleted instead of kept."""
    def fake_repo(repo_id: str, dest: str) -> str:
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "pytorch_model.bin").write_bytes(b"x" * 100)
        return dest

    monkeypatch.setattr(download_cli.hf_api, "download_repo", fake_repo)
    entry = _entry("hf_repo", hf_repo="org/repo", max_bytes=10)
    with pytest.raises(HubDownloadError, match="exceeds"):
        download_cli.download_entry(entry, str(tmp_path))
    assert not tmp_path.exists()


def test_snapshot_under_cap_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An under-cap snapshot keeps its honest unverified verdict."""
    def fake_repo(repo_id: str, dest: str) -> str:
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "pytorch_model.bin").write_bytes(b"x" * 100)
        return dest

    monkeypatch.setattr(download_cli.hf_api, "download_repo", fake_repo)
    entry = _entry("hf_repo", hf_repo="org/repo", size_bytes=100)
    report = download_cli.download_entry(entry, str(tmp_path))
    assert report["status"] == "unverified"
    assert tmp_path.exists()
