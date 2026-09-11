"""Hub download worker routing, verification, and cancellation."""

import asyncio
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from spikeforge_hub import download_cli, downloads, hf_api, verify
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import (
    HubDownloadCancelledError,
    HubDownloadError,
    HubExtraMissingError,
)


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


def test_url_source_fetches_and_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A URL entry is fetched, then its checksum and size are verified."""
    blob = b"payload"
    calls: List[str] = []

    def fake_fetch(url: str, path: str) -> str:
        calls.append(url)
        Path(path).write_bytes(blob)
        return path

    monkeypatch.setattr(download_cli, "fetch_url", fake_fetch)
    entry = _entry(
        "url", url="https://example.test/model.bin", sha256=_sha(blob),
        size_bytes=len(blob),
    )
    report = download_cli.download_entry(entry, str(tmp_path))
    assert calls == ["https://example.test/model.bin"]
    assert report["status"] == verify.VERIFIED


def test_url_checksum_mismatch_reports_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A tampered URL payload fails checksum verification."""
    def fake_fetch(url: str, path: str) -> str:
        Path(path).write_bytes(b"tampered")
        return path

    monkeypatch.setattr(download_cli, "fetch_url", fake_fetch)
    entry = _entry("url", url="https://example.test/x", sha256="0" * 64)
    report = download_cli.download_entry(entry, str(tmp_path))
    assert report["status"] == verify.FAILED


def test_bundled_source_never_touches_the_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bundled entry needs no fetch and is honestly unverified."""
    def boom(url: str, path: str) -> str:
        raise AssertionError("network used for a bundled entry")

    monkeypatch.setattr(download_cli, "fetch_url", boom)
    report = download_cli.download_entry(_entry("bundled"), str(tmp_path))
    assert report["status"] == verify.UNVERIFIED
    assert report["verified"] is False


def test_hf_source_uses_the_isolated_wrapper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An HF entry downloads through the isolated wrapper."""
    calls: List[str] = []

    def fake_repo(repo_id: str, dest: str) -> str:
        calls.append(repo_id)
        return dest

    monkeypatch.setattr(download_cli.hf_api, "download_repo", fake_repo)
    entry = _entry("hf_repo", hf_repo="org/repo")
    report = download_cli.download_entry(entry, str(tmp_path))
    assert calls == ["org/repo"]
    assert report["status"] == verify.UNVERIFIED


def test_hf_wrapper_names_the_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper raises a typed error naming the ``hub`` extra."""
    monkeypatch.setattr(hf_api.probe, "module", lambda: None)
    with pytest.raises(HubExtraMissingError):
        hf_api.download_repo("org/repo", "/tmp/hub-test")


def test_main_exits_nonzero_on_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """The worker exits non-zero when a seeded checksum is wrong."""
    def fake_fetch(url: str, path: str) -> str:
        Path(path).write_bytes(b"tampered")
        return path

    entry = _entry("url", url="https://example.test/x", sha256="0" * 64)
    monkeypatch.setattr(download_cli, "fetch_url", fake_fetch)
    monkeypatch.setattr(download_cli, "catalog_get", lambda _id: entry)
    monkeypatch.setattr(
        download_cli.cache, "entry_dir", lambda _id: str(tmp_path)
    )
    assert download_cli.main(["prog", "nir/example"]) == 1
    assert '"failed"' in capsys.readouterr().out


def test_main_reports_unknown_entry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """An unknown id fails cleanly instead of raising."""
    monkeypatch.setattr(download_cli, "catalog_get", lambda _id: None)
    assert download_cli.main(["prog", "nope"]) == 1
    assert "unknown catalog entry" in capsys.readouterr().out


def test_snapshot_exposes_progress_fields() -> None:
    """The idle snapshot carries the documented hub download keys."""
    manager = downloads.HubDownloadManager()
    assert manager.snapshot() == {
        "id": "",
        "status": "idle",
        "bytes": 0,
        "total_bytes": None,
        "verified": False,
    }


class _FakeProcess:
    """Minimal subprocess stand-in for the manager's terminal paths."""

    def __init__(
        self, returncode: Optional[int] = None, out: str = ""
    ) -> None:
        """Record the exit code and captured stdout the manager reads."""
        self.returncode = returncode
        self.stdout = io.StringIO(out)
        self.terminated = False

    def poll(self) -> Optional[int]:
        """Return the recorded exit code (``None`` while running)."""
        return self.returncode

    def terminate(self) -> None:
        """Record termination and a signal-like exit code."""
        self.terminated = True
        self.returncode = -15


def test_manager_cancel_terminates_and_reports_cancelled() -> None:
    """Cancelling terminates the child and reports ``cancelled``."""
    manager = downloads.HubDownloadManager()
    manager._process = _FakeProcess()
    manager._status = "downloading"
    manager.cancel()
    assert manager._process is not None
    assert manager._process.terminated is True
    events: List[Dict[str, Any]] = []

    async def emit(snapshot: Dict[str, Any]) -> None:
        events.append(snapshot)

    with pytest.raises(HubDownloadCancelledError):
        asyncio.run(manager._finish(emit))
    assert manager.snapshot()["status"] == "cancelled"
    assert events[-1]["status"] == "cancelled"


def test_manager_finish_marks_done_and_verified() -> None:
    """A successful worker exit resolves as ``done`` with verification."""
    manager = downloads.HubDownloadManager()
    manager._entry_id = "nir/example"
    manager._status = "downloading"
    manager._process = _FakeProcess(0, '{"verified": true}')
    events: List[Dict[str, Any]] = []

    async def emit(snapshot: Dict[str, Any]) -> None:
        events.append(snapshot)

    asyncio.run(manager._finish(emit))
    assert manager.snapshot()["status"] == "done"
    assert manager.snapshot()["verified"] is True


def test_manager_finish_marks_error_on_nonzero_exit() -> None:
    """A non-zero worker exit resolves as ``error`` and raises."""
    manager = downloads.HubDownloadManager()
    manager._entry_id = "nir/example"
    manager._status = "downloading"
    manager._process = _FakeProcess(1)

    async def emit(snapshot: Dict[str, Any]) -> None:
        pass

    with pytest.raises(HubDownloadError):
        asyncio.run(manager._finish(emit))
    assert manager.snapshot()["status"] == "error"


def test_manager_ensure_rejects_unknown_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ensure`` refuses an id that is not in the catalog."""
    manager = downloads.HubDownloadManager()
    monkeypatch.setattr(downloads, "catalog_get", lambda _id: None)

    async def emit(snapshot: Dict[str, Any]) -> None:
        pass

    with pytest.raises(HubDownloadError):
        asyncio.run(manager.ensure("nope", emit))
