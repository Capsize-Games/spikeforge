"""Download one hub entry in an isolated child process.

Run as ``python -m spikeforge_hub.download_cli <entry-id>`` so the parent
manager can terminate the process to cancel an in-flight download while
keeping the FastAPI event loop responsive. The worker prints one JSON line
describing the verified result; a non-zero exit means a published checksum or
size did not match.

Only ``https://`` URLs are fetched, every fetch carries a timeout, and no
fetch or snapshot may grow past its byte cap, so a catalog entry can neither
point at local files nor fill the disk.
"""

import contextlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from spikeforge_hub import cache, hf_api, verify
from spikeforge_hub.catalog import get as catalog_get
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubDownloadError, HubError

#: Seconds to wait for a hub fetch's connect and reads before giving up.
FETCH_TIMEOUT_SECONDS = 30

#: Default ceiling for one fetched artifact or snapshot: 2 GiB. An entry
#: may narrow it with ``max_bytes``; nothing may widen it past this.
DEFAULT_MAX_BYTES = 2 * 1024**3

#: Read size used while streaming a fetch, so large artifacts stream.
_CHUNK = 1 << 16

#: URL schemes ``fetch_url`` accepts; anything else is refused outright.
_ALLOWED_SCHEMES = ("https",)


def _filename(url: str) -> str:
    """Return a safe local filename for a download URL."""
    name = os.path.basename(url.split("?", 1)[0].rstrip("/"))
    return name or "artifact.bin"


def _allowed(url: str) -> bool:
    """Return True when ``url`` is an https URL."""
    try:
        scheme = urllib.parse.urlsplit(url).scheme.lower()
    except ValueError:
        return False
    return scheme in _ALLOWED_SCHEMES


def _require_https(url: str) -> None:
    """Refuse any fetch URL outside the https allowlist.

    Runs before any network or file access, so a refused URL touches
    nothing -- no socket, no redirect, no local path.
    """
    if not _allowed(url):
        raise HubDownloadError(
            url,
            "only https:// URLs are fetched; "
            "file:// and http:// are refused",
        )


class _HttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirect hops that leave the https allowlist."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Optional[urllib.request.Request]:
        """Follow https redirects; refuse hops off the allowlist."""
        if not _allowed(newurl):
            raise HubDownloadError(
                newurl, "redirect to a non-https URL is refused"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener() -> urllib.request.OpenerDirector:
    """Return an opener that refuses non-https redirect hops."""
    return urllib.request.build_opener(_HttpsRedirectHandler)


def _declared_size(response: Any) -> Optional[int]:
    """Return the response's Content-Length, or None when unparseable."""
    get = getattr(response, "getheader", None)
    raw = get("Content-Length") if callable(get) else None
    if raw is None:
        headers = getattr(response, "headers", None)
        get_header = getattr(headers, "get", None)
        if callable(get_header):
            raw = get_header("Content-Length")
    try:
        return int(str(raw)) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _refuse_when_declared_over(
    url: str, response: Any, cap: int
) -> None:
    """Raise before reading when Content-Length already exceeds ``cap``."""
    declared = _declared_size(response)
    if declared is not None and declared > cap:
        raise HubDownloadError(
            url, f"declared size {declared} exceeds the {cap} byte cap"
        )


def _drop_partial(path: str) -> None:
    """Delete a partial download, ignoring removal failures."""
    with contextlib.suppress(OSError):
        os.remove(path)


def _copy_capped(response: Any, path: str, cap: int, url: str) -> None:
    """Stream ``response`` to ``path``, refusing bytes past ``cap``."""
    total = 0
    try:
        with open(path, "wb") as handle:
            reads = iter(lambda: response.read(_CHUNK), b"")
            for block in reads:
                total += len(block)
                if total > cap:
                    raise HubDownloadError(
                        url, f"download exceeds the {cap} byte cap"
                    )
                handle.write(block)
    except (HubDownloadError, OSError):
        _drop_partial(path)
        raise


def _open(
    request: urllib.request.Request, url: str, timeout: int
) -> Any:
    """Open the guarded request, wrapping network failures."""
    try:
        return _opener().open(request, timeout=timeout)
    except (OSError, ValueError) as exc:
        raise HubDownloadError(
            url, f"network fetch failed: {exc}"
        ) from exc


def fetch_url(
    url: str,
    path: str,
    timeout: int = FETCH_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> str:
    """Download ``url`` to ``path`` and return the local path.

    Only ``https://`` is fetched; anything else is refused before any
    bytes move. The fetch aborts past ``timeout`` or ``max_bytes``.
    """
    _require_https(url)
    request = urllib.request.Request(url, method="GET")
    with _open(request, url, timeout) as response:
        _refuse_when_declared_over(url, response, max_bytes)
        _copy_capped(response, path, max_bytes, url)
    return path


def _cap(entry: HubEntry) -> int:
    """Return the entry's byte cap, defaulting to 2 GiB."""
    limit = entry.max_bytes
    if isinstance(limit, bool) or not isinstance(limit, int):
        return DEFAULT_MAX_BYTES
    return limit if limit > 0 else DEFAULT_MAX_BYTES


def download_entry(entry: HubEntry, dest: str) -> Dict[str, Any]:
    """Fetch ``entry`` into ``dest`` and report its verification outcome."""
    cache.ensure_dir(dest)
    if entry.source == "bundled":
        return _report(entry, dest, _bundled_result())
    if entry.source == "hf_repo":
        return _download_repo(entry, dest)
    if entry.source == "community":
        return _download_community(entry, dest)
    return _download_url(entry, dest)


def _download_repo(entry: HubEntry, dest: str) -> Dict[str, Any]:
    """Download a Hugging Face snapshot and verify its summed size.

    An over-cap snapshot is deleted and refused instead of kept.
    """
    hf_api.download_repo(entry.hf_repo or "", dest)
    _refuse_when_dir_over(entry, dest)
    return _report(entry, dest, verify.verify_dir(dest, entry.size_bytes))


def _refuse_when_dir_over(entry: HubEntry, dest: str) -> None:
    """Delete an over-cap snapshot and refuse it loudly."""
    total = cache.dir_bytes(dest)
    limit = _cap(entry)
    if total > limit:
        shutil.rmtree(dest, ignore_errors=True)
        raise HubDownloadError(
            entry.id,
            f"snapshot size {total} exceeds the {limit} byte cap",
        )


def _download_community(entry: HubEntry, dest: str) -> Dict[str, Any]:
    """Download a hosted community upload through its index URL.

    The index resolves the entry to content-addressed bytes; from there
    the fetch and the checksum/size verification are exactly a direct-URL
    download. The trust question is answered by the entry's own label
    (see ``CURATION.md``), not by this function.
    """
    return _download_url(entry, dest)


def _download_url(entry: HubEntry, dest: str) -> Dict[str, Any]:
    """Download a direct URL and verify its checksum and size."""
    url = entry.url or ""
    path = fetch_url(
        url, os.path.join(dest, _filename(url)), max_bytes=_cap(entry)
    )
    result = verify.verify_file(path, entry.sha256, entry.size_bytes)
    return _report(entry, dest, result)


def _bundled_result() -> verify.VerificationResult:
    """Return the honest verdict for a bundled, network-free entry."""
    return verify.VerificationResult(
        verify.UNVERIFIED,
        "bundled artifact; materialization is deferred to import",
        None,
        0,
    )


def _report(
    entry: HubEntry, dest: str, result: verify.VerificationResult
) -> Dict[str, Any]:
    """Build the JSON-able worker report for one entry."""
    return {
        "id": entry.id,
        "source": entry.source,
        "path": dest,
        "status": result.status,
        "verified": result.verified,
        "reason": result.reason,
        "sha256": result.sha256,
        "size_bytes": result.size_bytes,
    }


def _fail(reason: str, entry_id: str = "") -> int:
    """Print an error report and return the worker's failure code."""
    print(json.dumps({"id": entry_id, "status": "error", "reason": reason}))
    return 1


def main(argv: List[str]) -> int:
    """Download the entry named on the command line, then exit."""
    if len(argv) < 2:
        return _fail("missing entry id")
    entry = catalog_get(argv[1])
    if entry is None:
        return _fail("unknown catalog entry", argv[1])
    try:
        report = download_entry(entry, cache.entry_dir(entry.id))
    except HubError as exc:
        return _fail(str(exc), entry.id)
    print(json.dumps(report))
    return 1 if report["status"] == verify.FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
