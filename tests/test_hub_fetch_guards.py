"""Guards on the hub fetch path: scheme, timeout, and size cap."""

import hashlib
import socket
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from spikeforge_hub import download_cli
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubCatalogError, HubDownloadError


class _FakeResponse:
    """Minimal urlopen result: headers, chunked body, context manager."""

    def __init__(
        self, body: bytes, length: Optional[str] = None
    ) -> None:
        """Seed the body bytes and the declared Content-Length."""
        self._body = body
        self._length = length
        self.reads = 0

    def getheader(self, name: str) -> Optional[str]:
        """Return the declared Content-Length, else None."""
        if name.lower() == "content-length":
            return self._length
        return None

    def read(self, size: int = -1) -> bytes:
        """Return up to ``size`` body bytes, tracking read calls."""
        self.reads += 1
        if size is None or size < 0:
            chunk, self._body = self._body, b""
            return chunk
        chunk, self._body = self._body[:size], self._body[size:]
        return chunk

    def __enter__(self) -> "_FakeResponse":
        """Return this response for the ``with`` block."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Close the response (nothing to release)."""


class _FakeOpener:
    """Stand-in for the guarded opener, recording its timeout."""

    def __init__(self, response: Any) -> None:
        """Serve ``response`` (or raise it when it is an error)."""
        self._response = response
        self.timeouts: List[Any] = []

    def open(self, request: Any, timeout: Any = None) -> Any:
        """Record the timeout, then serve or raise the response."""
        self.timeouts.append(timeout)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _serve(
    monkeypatch: pytest.MonkeyPatch, response: Any
) -> _FakeOpener:
    """Serve ``response`` for every fetch and return the opener."""
    opener = _FakeOpener(response)
    monkeypatch.setattr(download_cli, "_opener", lambda: opener)
    return opener


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


def test_file_url_refused_without_touching_disk(tmp_path: Path) -> None:
    """A file:// URL is refused and no local bytes are copied."""
    secret = tmp_path / "secret.txt"
    secret.write_text("local-only")
    dest = str(tmp_path / "artifact.bin")
    with pytest.raises(HubDownloadError):
        download_cli.fetch_url(f"file://{secret}", dest)
    assert not Path(dest).exists()


def test_http_url_refused_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An http:// URL is refused before any socket opens."""
    def boom() -> Any:
        raise AssertionError("network used for a refused URL")

    monkeypatch.setattr(download_cli, "_opener", boom)
    with pytest.raises(HubDownloadError):
        download_cli.fetch_url(
            "http://169.254.169.254/latest/meta-data",
            str(tmp_path / "artifact.bin"),
        )


def test_unknown_scheme_refused(tmp_path: Path) -> None:
    """An ftp:// URL is refused like every other non-https URL."""
    with pytest.raises(HubDownloadError):
        download_cli.fetch_url(
            "ftp://example.test/model.bin",
            str(tmp_path / "artifact.bin"),
        )


def test_empty_url_refused_as_hub_error(tmp_path: Path) -> None:
    """A missing URL fails as a typed error, not a traceback."""
    with pytest.raises(HubDownloadError):
        download_cli.fetch_url("", str(tmp_path / "artifact.bin"))


@pytest.mark.parametrize(
    "target", ["file:///etc/passwd", "http://example.test/late"]
)
def test_redirect_off_allowlist_refused(target: str) -> None:
    """A redirect hop to file:// or http:// is refused loudly."""
    handler = download_cli._HttpsRedirectHandler()
    request = urllib.request.Request("https://example.test/first")
    with pytest.raises(HubDownloadError):
        handler.redirect_request(request, None, 302, "Found", {}, target)


def test_redirect_within_allowlist_followed() -> None:
    """A redirect hop to https:// is still followed."""
    handler = download_cli._HttpsRedirectHandler()
    request = urllib.request.Request("https://example.test/first")
    result = handler.redirect_request(
        request, None, 302, "Found", {}, "https://cdn.test/second"
    )
    assert isinstance(result, urllib.request.Request)
    assert result.full_url == "https://cdn.test/second"


def test_declared_size_over_cap_refused_before_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 10GB Content-Length aborts before a byte is read."""
    response = _FakeResponse(b"x", length=str(10**10))
    _serve(monkeypatch, response)
    dest = str(tmp_path / "artifact.bin")
    with pytest.raises(HubDownloadError):
        download_cli.fetch_url("https://example.test/big", dest)
    assert response.reads == 0
    assert not Path(dest).exists()


def test_streaming_cap_enforced_and_partial_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bytes past the cap abort the fetch and delete the partial."""
    _serve(monkeypatch, _FakeResponse(b"x" * 100))
    dest = str(tmp_path / "artifact.bin")
    with pytest.raises(HubDownloadError):
        download_cli.fetch_url(
            "https://example.test/drip", dest, max_bytes=10
        )
    assert not Path(dest).exists()


def test_small_https_download_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A normal small https download still succeeds and verifies."""
    blob = b"payload"
    opener = _serve(monkeypatch, _FakeResponse(blob, length=str(len(blob))))
    dest = str(tmp_path / "artifact.bin")
    assert download_cli.fetch_url("https://example.test/m", dest) == dest
    assert Path(dest).read_bytes() == blob
    assert opener.timeouts == [download_cli.FETCH_TIMEOUT_SECONDS]


def test_custom_timeout_reaches_the_opener(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit timeout is passed through to the opener."""
    opener = _serve(monkeypatch, _FakeResponse(b"ok"))
    download_cli.fetch_url(
        "https://example.test/m", str(tmp_path / "a.bin"), timeout=5
    )
    assert opener.timeouts == [5]


def test_network_timeout_wrapped_as_hub_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A socket timeout fails as a typed fetch error."""
    _serve(monkeypatch, socket.timeout("timed out"))
    with pytest.raises(HubDownloadError, match="network fetch failed"):
        download_cli.fetch_url(
            "https://example.test/slow", str(tmp_path / "a.bin")
        )


def test_garbage_content_length_streams_normally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unparseable Content-Length falls back to streaming."""
    blob = b"payload"
    _serve(monkeypatch, _FakeResponse(blob, length="bogus"))
    dest = str(tmp_path / "artifact.bin")
    download_cli.fetch_url("https://example.test/m", dest)
    assert Path(dest).read_bytes() == blob


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
