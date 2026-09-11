"""Download one hub entry in an isolated child process.

Run as ``python -m spikeforge_hub.download_cli <entry-id>`` so the parent
manager can terminate the process to cancel an in-flight download while
keeping the FastAPI event loop responsive. The worker prints one JSON line
describing the verified result; a non-zero exit means a published checksum or
size did not match.
"""

import json
import os
import shutil
import sys
import urllib.request
from typing import Any, Dict, List

from spikeforge_hub import cache, hf_api, verify
from spikeforge_hub.catalog import get as catalog_get
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubDownloadError, HubError


def _filename(url: str) -> str:
    """Return a safe local filename for a download URL."""
    name = os.path.basename(url.split("?", 1)[0].rstrip("/"))
    return name or "artifact.bin"


def fetch_url(url: str, path: str) -> str:
    """Download ``url`` to ``path`` and return the local path."""
    try:
        with urllib.request.urlopen(url) as response, open(path, "wb") as f:
            shutil.copyfileobj(response, f)
    except OSError as exc:
        raise HubDownloadError(url, f"network fetch failed: {exc}") from exc
    return path


def download_entry(entry: HubEntry, dest: str) -> Dict[str, Any]:
    """Fetch ``entry`` into ``dest`` and report its verification outcome."""
    cache.ensure_dir(dest)
    if entry.source == "bundled":
        return _report(entry, dest, _bundled_result())
    if entry.source == "hf_repo":
        return _download_repo(entry, dest)
    return _download_url(entry, dest)


def _download_repo(entry: HubEntry, dest: str) -> Dict[str, Any]:
    """Download a Hugging Face snapshot and verify its summed size."""
    hf_api.download_repo(entry.hf_repo or "", dest)
    return _report(entry, dest, verify.verify_dir(dest, entry.size_bytes))


def _download_url(entry: HubEntry, dest: str) -> Dict[str, Any]:
    """Download a direct URL and verify its checksum and size."""
    url = entry.url or ""
    path = fetch_url(url, os.path.join(dest, _filename(url)))
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
