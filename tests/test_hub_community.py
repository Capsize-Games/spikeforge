"""The ``community`` source and the hosted-index client."""

import contextlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest

from spikeforge_hub import cache, cli, community, download_cli, verify
from spikeforge_hub.catalog import (
    SCHEMA_VERSION,
    get,
    issues,
    list_entries,
    search,
)
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubCatalogError, HubDownloadError

_ENTRY: Dict[str, Any] = {
    "id": "@ada/mnist-fc",
    "name": "mnist-fc",
    "framework": "spikeforge",
    "kind": "bundle",
    "source": "community",
    "license": "BSD-3-Clause",
    "notes": "Community upload, version 1. Checks: unchecked.",
    "url": "https://cdn.spikeforge.net/0123abcd",
    "size_bytes": 7,
}


@contextlib.contextmanager
def _respond(
    body: bytes = b"", status: int = 200, etag: Optional[str] = None
) -> Iterator[SimpleNamespace]:
    """Yield a minimal ``urlopen`` response stand-in."""
    headers = {"ETag": etag} if etag else {}
    yield SimpleNamespace(status=status, read=lambda: body, headers=headers)


def _offline(request: Any) -> Any:
    """Pretend the network is unreachable."""
    raise urllib.error.URLError("no route to host")


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the hub cache into a temporary directory."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path / "hub"))
    return tmp_path


def _seed(root: Path, entries: List[Dict[str, Any]]) -> Path:
    """Write ``entries`` as the cached hosted index under ``root``."""
    dest = root / "hub" / community.INDEX_DIR_NAME
    dest.mkdir(parents=True)
    path = dest / community.INDEX_FILE_NAME
    payload = {"version": SCHEMA_VERSION, "entries": entries}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _document(entries: List[Dict[str, Any]]) -> bytes:
    """Return ``entries`` as an index document's encoded bytes."""
    payload = {"version": SCHEMA_VERSION, "entries": entries}
    return json.dumps(payload).encode("utf-8")


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    respond: Callable[[Any], Any],
) -> List[Any]:
    """Route ``urlopen`` to ``respond``, recording every request."""
    seen: List[Any] = []

    def fake_open(request: Any, timeout: int = 0) -> Any:
        seen.append(request)
        return respond(request)

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    return seen


def test_community_source_validates_with_an_at_id() -> None:
    """A hosted ``@handle/slug`` entry passes schema validation."""
    entry = HubEntry.from_dict(dict(_ENTRY))
    assert entry.source == "community"
    assert entry.id == "@ada/mnist-fc"


def test_community_source_requires_its_url() -> None:
    """A community entry without a ``url`` is rejected by name."""
    data = dict(_ENTRY)
    del data["url"]
    with pytest.raises(HubCatalogError, match="needs a 'url'"):
        HubEntry.from_dict(data)


def test_at_sign_is_only_allowed_leading() -> None:
    """A bare or mid-id ``@`` is rejected; curated ids are untouched."""
    for bad in ("@", "nir/fc@small", "@@ada/x"):
        with pytest.raises(HubCatalogError, match="id must match"):
            HubEntry.from_dict({**_ENTRY, "id": bad})
    assert HubEntry.from_dict({**_ENTRY, "id": "nir/x"}).id == "nir/x"


def test_fetch_stores_the_index_and_its_etag(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200 stores the bytes and ETag, and reports a fresh namespace."""
    body = _document([_ENTRY])
    seen = _serve(
        monkeypatch, lambda req: _respond(body, etag='"v1"')
    )
    result = community.fetch_index(base="https://hub.test")
    assert result.refreshed is True
    assert [entry.id for entry in result.entries] == ["@ada/mnist-fc"]
    assert result.issues == []
    assert seen[0].full_url == "https://hub.test/v1/index.json"
    assert "If-None-Match" not in seen[0].headers
    assert community.read_etag() == '"v1"'


def test_fetch_revalidates_and_keeps_cache_on_304(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored ETag is sent back; a 304 loads the cache, not the wire."""
    path = _seed(sandbox, [_ENTRY])
    before = path.read_bytes()
    (path.parent / community.ETAG_FILE_NAME).write_text('"v1"')
    seen = _serve(monkeypatch, lambda req: _respond(status=304))
    result = community.fetch_index(base="https://hub.test")
    sent = dict(seen[0].header_items())
    assert sent["If-none-match"] == '"v1"'
    assert result.refreshed is False
    assert [entry.id for entry in result.entries] == ["@ada/mnist-fc"]
    assert path.read_bytes() == before


def test_fetch_falls_back_to_cache_when_offline(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead network with a cache returns stale bytes and says so."""
    _seed(sandbox, [_ENTRY])

    _serve(monkeypatch, _offline)
    result = community.fetch_index(base="https://hub.test")
    assert [entry.id for entry in result.entries] == ["@ada/mnist-fc"]
    assert result.refreshed is False
    assert any("stale" in issue for issue in result.issues)


def test_fetch_without_cache_or_network_raises(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing cached and nothing reachable is a named failure."""
    _serve(monkeypatch, _offline)
    with pytest.raises(HubDownloadError, match="nothing is cached"):
        community.fetch_index(base="https://hub.test")


def test_malformed_document_keeps_the_previous_cache(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server garbage never clobbers a good cache; the old bytes win."""
    path = _seed(sandbox, [_ENTRY])
    before = path.read_bytes()
    _serve(monkeypatch, lambda req: _respond(b"{nope"))
    result = community.fetch_index(base="https://hub.test")
    assert result.refreshed is False
    assert [entry.id for entry in result.entries] == ["@ada/mnist-fc"]
    assert path.read_bytes() == before


def test_malformed_hosted_entry_is_reported_not_dropped(
    sandbox: Path,
) -> None:
    """A bad hosted record lands in issues while its siblings load."""
    bad = dict(_ENTRY, id="@mallory/x", source="nope")
    _seed(sandbox, [_ENTRY, bad])
    entries, issues = community.load_cached()
    assert [entry.id for entry in entries] == ["@ada/mnist-fc"]
    assert len(issues) == 1
    assert "@mallory/x" in issues[0]


def test_catalog_renders_community_ids_alongside_curated(
    sandbox: Path,
) -> None:
    """A cached community id resolves, lists, and searches like any id."""
    _seed(sandbox, [_ENTRY])
    assert get("@ada/mnist-fc") is not None
    assert get("nir/fc_small") is not None
    assert "@ada/mnist-fc" in [
        card["id"] for card in list_entries()
    ]
    assert search("@ada/mnist-fc")


def test_corrupt_cache_is_an_issue_not_an_outage(
    sandbox: Path,
) -> None:
    """An unreadable community file is reported; curation survives."""
    dest = sandbox / "hub" / community.INDEX_DIR_NAME
    dest.mkdir(parents=True)
    (dest / community.INDEX_FILE_NAME).write_text("{nope", encoding="utf-8")
    assert get("nir/fc_small") is not None
    assert any(
        "community index unavailable" in issue
        for issue in issues()
    )


def test_list_includes_cached_community_entries(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``spikeforge-hub list`` renders the merged namespace as JSON."""
    _seed(sandbox, [_ENTRY])
    assert cli.main(["list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "@ada/mnist-fc" in [item["id"] for item in payload["entries"]]
    assert payload["issues"] == []


def test_community_download_fetches_its_index_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A community entry downloads its ``url`` with verification."""
    blob = b"bundle-bytes"
    calls: List[str] = []

    def fake_fetch(url: str, path: str) -> str:
        calls.append(url)
        Path(path).write_bytes(blob)
        return path

    monkeypatch.setattr(download_cli, "fetch_url", fake_fetch)
    entry = HubEntry.from_dict({**_ENTRY, "size_bytes": len(blob)})
    report = download_cli.download_entry(entry, str(tmp_path))
    assert calls == ["https://cdn.spikeforge.net/0123abcd"]
    assert report["source"] == "community"
    assert report["status"] == verify.UNVERIFIED
