"""Fetch and cache the hosted community catalog index.

``GET /v1/index.json`` on the hub API renders the published community
namespace in this project's own catalog schema, so the bytes are cached
under :data:`spikeforge.config.HUB_CACHE_DIR` and then loaded through the
existing :func:`spikeforge_hub.catalog.load_catalog` -- no second schema,
no second validator. A malformed hosted entry is reported through the
returned issues, exactly like a malformed curated one.

Conditional requests keep refreshes cheap: the server's ``ETag`` is stored
alongside the cached document and sent back as ``If-None-Match``. When the
network is unreachable but a cached copy exists, the cached copy is used
and the staleness is reported as an issue rather than hidden.
"""

import contextlib
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from spikeforge.config import HUB_API_BASE
from spikeforge_hub import cache
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubCatalogError, HubDownloadError

#: Subdirectory of the hub cache holding the hosted index. No catalog id
#: can slug to it: entry directories never start with a dot.
INDEX_DIR_NAME = ".community"
#: Cached hosted-index document, in the shipped catalog's schema.
INDEX_FILE_NAME = "index.json"
#: The server's ETag for the cached document, when it sent one.
ETAG_FILE_NAME = "index.etag"
#: Seconds to wait for the hosted index before giving up.
FETCH_TIMEOUT = 30


@dataclass(frozen=True)
class FetchResult:
    """The community namespace as loaded, and whether it is fresh."""

    entries: List[HubEntry] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    #: True when the bytes came from the network just now; False when they
    #: are the previously cached copy (unchanged per ETag, or the network
    #: failed and the cache covered for it).
    refreshed: bool = False


def _dir() -> str:
    """Return the hosted-index cache directory without creating it."""
    return os.path.join(cache.cache_root(), INDEX_DIR_NAME)


def index_dir() -> str:
    """Return (and create) the cache directory for the hosted index."""
    return cache.ensure_dir(_dir())


def index_path() -> str:
    """Return the path of the cached hosted-index document."""
    return os.path.join(index_dir(), INDEX_FILE_NAME)


def etag_path() -> str:
    """Return the path of the stored index ETag, when the server sent one."""
    return os.path.join(index_dir(), ETAG_FILE_NAME)


def read_etag() -> Optional[str]:
    """Return the stored ETag, or ``None`` when there is none."""
    try:
        with open(
            os.path.join(_dir(), ETAG_FILE_NAME), encoding="utf-8"
        ) as handle:
            value = handle.read().strip()
    except OSError:
        return None
    return value or None


def load_cached() -> Tuple[List[HubEntry], List[str]]:
    """Load the cached hosted index, or return empty when never fetched.

    Read-only: merely listing the catalog must not create cache
    directories. Only a refresh that stores bytes creates them.
    """
    from spikeforge_hub.catalog import load_catalog

    path = Path(os.path.join(_dir(), INDEX_FILE_NAME))
    if not path.is_file():
        return [], []
    return load_catalog(path)


def fetch_index(
    base: Optional[str] = None, timeout: int = FETCH_TIMEOUT
) -> FetchResult:
    """Refresh the hosted index and return the community namespace.

    On a network failure the previously cached copy is used when one
    exists, with the staleness reported as an issue; with no cache and no
    network there is nothing to load, which raises.
    """
    url = (base or HUB_API_BASE).rstrip("/") + "/v1/index.json"
    try:
        return _fetch(url, timeout)
    except (HubDownloadError, HubCatalogError, OSError) as exc:
        return _fallback(exc)


def _fetch(url: str, timeout: int) -> FetchResult:
    """Return the namespace from the network, storing what it sent."""
    request = urllib.request.Request(url, method="GET")
    etag = read_etag()
    if etag is not None:
        request.add_header("If-None-Match", etag)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _handle(url, response)
    except urllib.error.HTTPError as exc:
        raise HubDownloadError(
            "community index", f"{url} answered HTTP {exc.code}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise HubDownloadError(
            "community index", f"cannot reach {url}: {exc}"
        ) from exc


def _handle(url: str, response: object) -> FetchResult:
    """Store a live response, or load the cache on ``304 Not Modified``."""
    status = getattr(response, "status", 200)
    if status == 304:
        entries, issues = load_cached()
        return FetchResult(list(entries), list(issues), refreshed=False)
    body = _body(url, response)
    _store(body, _response_etag(response))
    entries, issues = load_cached()
    return FetchResult(list(entries), list(issues), refreshed=True)


def _body(url: str, response: object) -> bytes:
    """Return the response body, refusing an empty document."""
    read = getattr(response, "read", None)
    body = read() if callable(read) else b""
    if not body:
        raise HubDownloadError(
            "community index", f"{url} returned an empty document"
        )
    return bytes(body)


def _response_etag(response: object) -> Optional[str]:
    """Return the response's ETag header, or ``None`` when absent."""
    headers = getattr(response, "headers", None)
    get = getattr(headers, "get", None)
    value = get("ETag") if callable(get) else None
    return str(value).strip() or None if value else None


def _store(body: bytes, etag: Optional[str]) -> None:
    """Validate the fetched bytes, then atomically replace the cache.

    Validation runs before the replace so a malformed server document can
    never clobber a good cached copy; the previous cache stays in place
    and the refresh simply fails loudly.
    """
    from spikeforge_hub.catalog import load_catalog

    path = index_path()
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        handle.write(body)
    try:
        load_catalog(Path(tmp))
    except HubCatalogError:
        _remove_quietly(tmp)
        raise
    os.replace(tmp, path)
    _remember_etag(etag)


def _remember_etag(etag: Optional[str]) -> None:
    """Store the server's ETag, or forget it when it sent none."""
    if etag is None:
        _drop_etag()
        return
    with open(etag_path(), "w", encoding="utf-8") as handle:
        handle.write(etag)


def _remove_quietly(path: str) -> None:
    """Delete ``path`` when it exists, ignoring removal failures."""
    with contextlib.suppress(OSError):
        os.remove(path)


def _drop_etag() -> None:
    """Forget the stored ETag when the server stopped sending one."""
    with contextlib.suppress(OSError):
        os.remove(etag_path())


def _fallback(exc: Exception) -> FetchResult:
    """Return the cached namespace, or raise when there is no cache."""
    try:
        entries, issues = load_cached()
    except (OSError, HubCatalogError) as cached_exc:
        raise HubDownloadError(
            "community index", f"refresh failed ({exc}); {cached_exc}"
        ) from exc
    if not entries and not issues and not _has_cache():
        raise HubDownloadError(
            "community index", f"refresh failed and nothing is cached: {exc}"
        ) from exc
    note = f"community index is stale: refresh failed ({exc})"
    return FetchResult(list(entries), [note, *issues], refreshed=False)


def _has_cache() -> bool:
    """Return True when an index document has been cached before."""
    return os.path.isfile(os.path.join(_dir(), INDEX_FILE_NAME))
