"""Push saved models to, and pull community models from, the hub API.

Push builds a ``.spkf`` with the same
:func:`spikeforge.serving.bundle.build` the server uses, then runs the
two-phase upload (reserve, send bytes, commit). Pull resolves the id
against a refreshed hosted index and runs the existing
inspect/compat/promote funnel -- the latest version through the catalog
entry, a pinned version through its bytes.

A bundle's ``expected_metrics`` is always an uploader claim: the push
flags (or, for a reference id, the catalog's measured row) fill it, the
verification runner never does -- it has the bytes but no test split.
Any display of these numbers must say uploader-claimed, never reproduced.
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
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubArtifactError, HubDownloadError
from spikeforge_hub.import_model import import_model
from spikeforge_hub.inspect import reference_path, resolve_path


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
    client: HubClient, name: str, version: str,
    license_id: Optional[str] = None,
    slug: Optional[str] = None, summary: str = "", dataset: str = "",
    dataset_license: str = "", dataset_attribution: str = "",
    description: str = "", checkpoint: Optional[str] = None,
    test_accuracy: Optional[float] = None,
    test_samples: Optional[int] = None, test_dataset: str = "",
) -> Dict[str, Any]:
    """Build ``name`` and publish it through ``client``.

    ``name`` is a saved model, a ``reference/*`` catalog id (whose
    packaged weights are staged, built, and pushed with the catalog's
    provenance), or -- with ``checkpoint`` -- any model-store-shaped
    ``.pt`` file. The slug defaults to the model name, or the catalog
    row for a reference id; the server lower-cases it and refuses what
    is not a valid slug. ``license_id`` defaults to the reference
    entry's licence for a reference id, and is otherwise required.
    """
    workdir = tempfile.mkdtemp(prefix="spikeforge-push-")
    try:
        staged, entry = _resolve_source(name, checkpoint)
        metrics = _metrics_for(
            name, entry, test_accuracy, test_samples, test_dataset
        )
        path = _build_artifact(name, workdir, staged, metrics)
        intent, data = _intent(
            version, _license_for(name, license_id, entry),
            path, _slug_for(name, slug, entry), summary,
            dataset, dataset_license, dataset_attribution,
            description, entry,
        )
        outcome = _push_bytes(client, intent, data)
        outcome["expected_metrics"] = metrics
        return outcome
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _resolve_source(
    name: str, checkpoint: Optional[str]
) -> Tuple[Optional[str], Optional[HubEntry]]:
    """Return ``(stage_from, entry)`` for the requested push bytes.

    ``stage_from`` is a ``.pt`` path to stage into the store before the
    build, or None to build a saved model directly. A store miss falls
    back to a ``reference/*`` catalog id; anything else is refused.
    """
    from spikeforge.network import model_store

    if checkpoint is not None:
        if not os.path.isfile(checkpoint):
            raise HubArtifactError(
                checkpoint, f"no checkpoint file at {checkpoint!r}"
            )
        return checkpoint, None
    saved = {entry["name"] for entry in model_store.list_models()}
    if name in saved:
        return None, None
    entry = catalog_get(name)
    if entry is not None and entry.source == "reference":
        return reference_path(entry), entry
    raise HubArtifactError(name, f"no saved model named {name!r}")


def _license_for(
    name: str, license_id: Optional[str], entry: Optional[HubEntry]
) -> str:
    """Return the push licence: explicit, reference, or refused."""
    if license_id:
        return license_id
    if entry is not None and entry.license:
        return entry.license
    raise HubArtifactError(
        name, "push needs a licence: pass --license"
    )


def _slug_for(
    name: str, slug: Optional[str], entry: Optional[HubEntry]
) -> str:
    """Return the push slug: explicit, reference row, or model name."""
    if slug:
        return slug
    if entry is not None:
        return entry.id.partition("/")[2] or name
    return name


def _check_accuracy(name: str, accuracy: Any) -> float:
    """Validate a claimed accuracy like the catalog validates scores."""
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        raise HubArtifactError(
            name, "test_accuracy must be a number"
        )
    if not 0.0 <= float(accuracy) <= 100.0:
        raise HubArtifactError(
            name,
            f"test_accuracy {accuracy} is outside 0-100 percent",
        )
    return float(accuracy)


def _check_samples(name: str, samples: Any) -> int:
    """Validate a claimed sample count like the catalog validates it."""
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise HubArtifactError(
            name, "test_samples must be an integer"
        )
    return samples


def _check_claim(
    name: str,
    accuracy: Optional[float],
    samples: Optional[int],
    dataset: str,
) -> Dict[str, Any]:
    """Validate an explicit accuracy claim, or return no claim.

    Samples and dataset qualify the accuracy, so they are refused
    without one; unstated stays ``{}``, never a zero that reads as a
    measurement.
    """
    if accuracy is None and samples is None and not dataset:
        return {}
    if accuracy is None:
        raise HubArtifactError(
            name, "--test-samples/--test-dataset need --test-accuracy"
        )
    claimed: Dict[str, Any] = {
        "test_accuracy": _check_accuracy(name, accuracy)
    }
    if samples is not None:
        claimed["test_samples"] = _check_samples(name, samples)
    if dataset:
        claimed["test_dataset"] = str(dataset)
    return claimed


def _reference_metrics(entry: HubEntry) -> Dict[str, Any]:
    """Return the catalog-measured scores for a reference push."""
    metrics: Dict[str, Any] = {}
    if entry.test_accuracy is not None:
        metrics["test_accuracy"] = float(entry.test_accuracy)
    if entry.test_samples is not None:
        metrics["test_samples"] = entry.test_samples
    if entry.dataset:
        metrics["test_dataset"] = entry.dataset
    return metrics


def _metrics_for(
    name: str,
    entry: Optional[HubEntry],
    accuracy: Optional[float],
    samples: Optional[int],
    dataset: str,
) -> Dict[str, Any]:
    """Return the ``expected_metrics`` the pushed bundle will carry.

    An explicit claim wins when given (filling an unclaimed dataset
    from the reference row); otherwise a reference push carries its
    catalog-measured scores, and anything else ships ``{}``.
    """
    claimed = _check_claim(name, accuracy, samples, dataset)
    if claimed:
        if (
            "test_dataset" not in claimed
            and entry is not None and entry.dataset
        ):
            claimed["test_dataset"] = entry.dataset
        return claimed
    if entry is not None:
        return _reference_metrics(entry)
    return {}


def _stage_name(slug: str) -> str:
    """Return a store name for staged push bytes that is free to use."""
    from spikeforge.network import model_store

    base = "push-staging-" + model_store.safe_name(slug)
    taken = {entry["name"] for entry in model_store.list_models()}
    name, counter = base, 2
    while name in taken:
        name, counter = f"{base}-{counter}", counter + 1
    return name


def _build_staged(
    name: str, workdir: str, stage_from: str,
    metrics: Dict[str, Any],
) -> str:
    """Stage ``stage_from`` into the store, build it, and clean up."""
    from spikeforge.network import model_store
    from spikeforge.serving import bundle as bundle_mod

    staged = _stage_name(name)
    shutil.copyfile(stage_from, model_store.path_for(staged))
    try:
        out = os.path.join(workdir, f"{staged}.spkf")
        try:
            bundle_mod.build(staged, out=out, expected_metrics=metrics)
        except Exception as exc:  # bundle failures must not escape untagged
            raise HubArtifactError(
                name, f"cannot build a bundle from {name!r}: {exc}"
            ) from exc
        return out
    finally:
        model_store.delete(staged)


def _build_artifact(
    name: str, workdir: str, stage_from: Optional[str],
    metrics: Dict[str, Any],
) -> str:
    """Build ``name`` into a ``.spkf`` inside ``workdir`` and return it."""
    from spikeforge.network import model_store
    from spikeforge.serving import bundle as bundle_mod

    if stage_from is not None:
        return _build_staged(name, workdir, stage_from, metrics)
    out = os.path.join(workdir, f"{model_store.safe_name(name)}.spkf")
    try:
        bundle_mod.build(name, out=out, expected_metrics=metrics)
    except Exception as exc:  # bundle failures must not escape untagged
        raise HubArtifactError(
            name, f"cannot build a bundle from {name!r}: {exc}"
        ) from exc
    return out


def _read_bytes(path: str) -> bytes:
    """Return the whole file at ``path`` as bytes."""
    with open(path, "rb") as handle:
        return handle.read()


def _provenance(
    entry: Optional[HubEntry], dataset: str, dataset_license: str,
    dataset_attribution: str,
) -> Tuple[str, str, str]:
    """Return the dataset declarations, defaulting to the reference row."""
    if entry is None:
        return dataset, dataset_license, dataset_attribution
    return (
        dataset or entry.dataset or "",
        dataset_license or entry.dataset_license or "",
        dataset_attribution or entry.dataset_attribution or "",
    )


def _intent(
    version: str, license_id: str, path: str, slug: str, summary: str,
    dataset: str, dataset_license: str, dataset_attribution: str,
    description: str, entry: Optional[HubEntry],
) -> Tuple[Dict[str, Any], bytes]:
    """Return the upload intent and the bytes it declares."""
    data = _read_bytes(path)
    declared = _provenance(
        entry, dataset, dataset_license, dataset_attribution
    )
    intent = {
        "name": slug,
        "version": version,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "license": license_id,
        "dataset": declared[0],
        "dataset_license": declared[1],
        "dataset_attribution": declared[2],
        "summary": summary,
        "description": description,
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


def _bundle_metrics(path: str) -> Optional[Dict[str, Any]]:
    """Return the uploader-claimed metrics in the bundle at ``path``.

    None when the bytes are not a readable bundle; ``{}`` when they are
    a bundle that states no claim. The numbers are the uploader's own --
    the verification runner never writes this field.
    """
    from spikeforge.serving.bundle import DeploymentBundle

    try:
        manifest = DeploymentBundle.load(path, strict=True).manifest
    except Exception:  # not a bundle; there is nothing to surface
        return None
    metrics = manifest.get("expected_metrics")
    if not isinstance(metrics, dict):
        return None
    return dict(metrics)


def _metrics_report(path: str) -> Dict[str, Any]:
    """Return the pull report's metrics block for the bytes at ``path``."""
    metrics = _bundle_metrics(path)
    return {
        "expected_metrics": metrics,
        "metrics_claim": "uploader-claimed" if metrics else None,
    }


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
    pulled = {
        "id": target.id,
        "version": None,
        "downloaded": report is not None,
        "download": report,
        "import": import_model(entry_id=entry.id),
        "index_refreshed": refreshed,
        "index_issues": list(index_issues),
    }
    pulled.update(_metrics_report(resolve_path(entry)))
    return pulled


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
    pulled = {
        "id": target.id,
        "version": target.version,
        "downloaded": downloaded,
        "download": _pinned_download(path),
        "import": import_model(path=path),
        "index_refreshed": refreshed,
        "index_issues": list(index_issues),
    }
    pulled.update(_metrics_report(path))
    return pulled


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
