"""Push saved models to, and pull community models from, the hub API.

Push builds a ``.spkf`` with the same
:func:`spikeforge.serving.bundle.build` the server uses, then runs the
two-phase upload (reserve, send bytes, commit). Pull resolves the id
against a refreshed hosted index and runs the existing
inspect/compat/promote funnel -- the latest version through the catalog
entry, a pinned version through its bytes.
"""

import hashlib
import os
import shutil
import tempfile
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from spikeforge.config import HUB_API_BASE
from spikeforge_hub import cache, download
from spikeforge_hub.catalog import get as catalog_get
from spikeforge_hub.client import HubClient
from spikeforge_hub.community import fetch_index
from spikeforge_hub.download_cli import fetch_url
from spikeforge_hub.errors import HubArtifactError, HubDownloadError
from spikeforge_hub.import_model import import_model


@dataclass(frozen=True)
class PullTarget:
    """A parsed ``@handle/slug[@version]`` pull request."""

    id: str
    handle: str
    slug: str
    version: Optional[str] = None


def parse_pull_id(text: str) -> PullTarget:
    """Split ``@handle/slug[@version]``, or refuse it with a reason."""
    raw = (text or "").strip()
    handle, slash, rest = (
        raw[1:].partition("/") if raw.startswith("@") else ("", "", "")
    )
    slug, at, version = rest.partition("@")
    if not raw.startswith("@") or not slash or not handle or not slug:
        raise HubDownloadError(
            text, "a pull id looks like `@handle/slug[@version]`"
        )
    if "/" in slug or "@" in version or "/" in version or (at and not version):
        raise HubDownloadError(raw, f"malformed pull id {raw!r}")
    return PullTarget(f"@{handle}/{slug}", handle, slug, version or None)


def push_saved_model(
    client: HubClient, name: str, version: str, license_id: str,
    slug: Optional[str] = None, summary: str = "", dataset: str = "",
    dataset_license: str = "", dataset_attribution: str = "",
) -> Dict[str, Any]:
    """Build saved model ``name`` and publish it through ``client``.

    The slug defaults to the model name; the server lower-cases it and
    refuses what is not a valid slug.
    """
    workdir = tempfile.mkdtemp(prefix="spikeforge-push-")
    try:
        path = _build_artifact(name, workdir)
        intent, data = _intent(
            name, version, license_id, path, slug, summary, dataset,
            dataset_license, dataset_attribution,
        )
        return _push_bytes(client, intent, data)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _build_artifact(name: str, workdir: str) -> str:
    """Build ``name`` into a ``.spkf`` inside ``workdir`` and return it."""
    from spikeforge.network import model_store
    from spikeforge.serving import bundle as bundle_mod

    saved = {entry["name"] for entry in model_store.list_models()}
    if name not in saved:
        raise HubArtifactError(name, f"no saved model named {name!r}")
    out = os.path.join(workdir, f"{model_store.safe_name(name)}.spkf")
    try:
        bundle_mod.build(name, out=out)
    except Exception as exc:  # bundle failures must not escape untagged
        raise HubArtifactError(
            name, f"cannot build a bundle from {name!r}: {exc}"
        ) from exc
    return out


def _read_bytes(path: str) -> bytes:
    """Return the whole file at ``path`` as bytes."""
    with open(path, "rb") as handle:
        return handle.read()


def _intent(
    name: str, version: str, license_id: str, path: str,
    slug: Optional[str], summary: str, dataset: str,
    dataset_license: str, dataset_attribution: str,
) -> Tuple[Dict[str, Any], bytes]:
    """Return the upload intent and the bytes it declares."""
    data = _read_bytes(path)
    intent = {
        "name": slug or name,
        "version": version,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "license": license_id,
        "dataset": dataset,
        "dataset_license": dataset_license,
        "dataset_attribution": dataset_attribution,
        "summary": summary,
    }
    return intent, data


def _push_bytes(
    client: HubClient, intent: Dict[str, Any], data: bytes
) -> Dict[str, Any]:
    """Run the reserve/send/commit upload and report its outcome."""
    created = client.create_upload(intent)
    upload_id = str(created["upload_id"])
    client.put_content(upload_id, data)
    committed = client.commit_upload(upload_id)
    return {
        "name": intent["name"],
        "version": intent["version"],
        "size_bytes": intent["size_bytes"],
        "sha256": intent["sha256"],
        "upload_id": upload_id,
        "state": committed["state"],
        "published_at": committed.get("published_at"),
    }


def pull_model(pull_id: str) -> Dict[str, Any]:
    """Download and import a community id through the funnel.

    The hosted index is refreshed first; when the network is down the
    previously cached index covers for it and the report says it is
    stale. A pinned version downloads its own bytes, since the index
    only carries the latest.
    """
    target = parse_pull_id(pull_id)
    fetched = fetch_index()
    if target.version is None:
        return _pull_latest(target, fetched.refreshed, fetched.issues)
    return _pull_pinned(target, fetched.refreshed, fetched.issues)


def _pull_latest(
    target: PullTarget, refreshed: bool, index_issues: List[str]
) -> Dict[str, Any]:
    """Pull the index's entry for ``target`` through the funnel."""
    entry = catalog_get(target.id)
    if entry is None:
        raise HubDownloadError(target.id, _unknown_hint(target, refreshed))
    if entry.source != "community":
        raise HubDownloadError(target.id, "pull needs a community id")
    report = None if cache.cached(entry.id) else download(target.id)
    return {
        "id": target.id,
        "version": None,
        "downloaded": report is not None,
        "download": report,
        "import": import_model(entry_id=entry.id),
        "index_refreshed": refreshed,
        "index_issues": list(index_issues),
    }


def _unknown_hint(target: PullTarget, refreshed: bool) -> str:
    """Explain an unresolvable pull id, naming a stale index when so."""
    if refreshed:
        return "unknown catalog entry"
    return (
        "unknown catalog entry; the index is stale (refresh failed), "
        "so this id may simply be newer than the cache"
    )


def _version_url(target: PullTarget) -> str:
    """Return the download URL for one pinned community version."""
    quote = urllib.parse.quote
    return (
        f"{HUB_API_BASE.rstrip('/')}/v1/models/{quote(target.handle)}/"
        f"{quote(target.slug)}/versions/{quote(target.version or '')}/"
        "download"
    )


def _pinned_filename(target: PullTarget) -> str:
    """Return the cache filename for one pinned version's bytes."""
    quoted = urllib.parse.quote(target.version or "", safe="")
    return f"{target.slug}-{quoted}.spkf"


def _pull_pinned(
    target: PullTarget, refreshed: bool, index_issues: List[str]
) -> Dict[str, Any]:
    """Pull one pinned version's bytes, then import them by path."""
    dest = cache.entry_dir(target.id)
    path = os.path.join(dest, _pinned_filename(target))
    downloaded = False
    if not os.path.isfile(path):
        fetch_url(_version_url(target), path)
        downloaded = True
    return {
        "id": target.id,
        "version": target.version,
        "downloaded": downloaded,
        "download": _pinned_download(path),
        "import": import_model(path=path),
        "index_refreshed": refreshed,
        "index_issues": list(index_issues),
    }


def _pinned_download(path: str) -> Dict[str, Any]:
    """Report a pinned fetch, whose bytes carry no checksum to check."""
    return {
        "path": path,
        "verified": False,
        "reason": (
            "a pinned version has no published checksum to verify "
            "against; bytes as served"
        ),
    }
