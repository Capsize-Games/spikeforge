"""Bundle rejection: integrity, format, compatibility, version gates."""

import json
import zipfile
from typing import Any, Dict, Mapping

import pytest
from _bundle_helpers import _written

from spikeforge.network import model_store
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.errors import (
    BundleCompatibilityError,
    BundleFormatError,
    BundleIntegrityError,
    BundleNotFoundError,
)
from spikeforge.topology import registry


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint reads and writes into a per-test directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _manifest(entries: Mapping[str, bytes]) -> Dict[str, Any]:
    """Return the decoded manifest of a bundle's entry mapping."""
    return json.loads(entries[bm.MANIFEST_NAME].decode("utf-8"))


def _with_manifest(
    entries: Mapping[str, bytes], **changes: Any
) -> Dict[str, bytes]:
    """Return ``entries`` with the manifest's top-level keys updated."""
    manifest = _manifest(entries)
    manifest.update(changes)
    return {**entries, bm.MANIFEST_NAME: bm.dump_json(manifest)}


def _with_torch(
    entries: Mapping[str, bytes], version: str
) -> Dict[str, bytes]:
    """Return ``entries`` with the recorded torch version replaced."""
    manifest = _manifest(entries)
    versions = dict(manifest.get("library_versions") or {})
    versions["torch"] = version
    manifest["library_versions"] = versions
    return {**entries, bm.MANIFEST_NAME: bm.dump_json(manifest)}


def _rewrite(path: str, mutate: Any, regenerate: bool = True) -> None:
    """Rewrite a bundle, optionally recomputing its SHA256SUMS."""
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries = mutate(entries)
    if regenerate:
        payloads = {
            name: payload
            for name, payload in entries.items()
            if name != bm.SUMS_NAME
        }
        entries[bm.SUMS_NAME] = bm.sums_text(payloads).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def test_rejects_tampered_weights(tmp_path: Any) -> None:
    """A byte changed after signing is an integrity error."""
    out = _written(tmp_path)
    _rewrite(
        out,
        lambda entries: {
            **entries,
            bm.WEIGHTS_NAME: entries[bm.WEIGHTS_NAME] + b"\x00",
        },
        regenerate=False,
    )
    with pytest.raises(BundleIntegrityError) as info:
        DeploymentBundle.load(out)
    assert info.value.entry == bm.WEIGHTS_NAME


def test_rejects_unknown_format_marker(tmp_path: Any) -> None:
    """A bundle whose marker is wrong is a format error."""
    out = _written(tmp_path)
    _rewrite(out, lambda entries: _with_manifest(entries, format="nope"))
    with pytest.raises(BundleFormatError):
        DeploymentBundle.load(out)


def test_rejects_missing_required_entry(tmp_path: Any) -> None:
    """A bundle missing a required entry is a format error."""
    out = _written(tmp_path)
    _rewrite(
        out,
        lambda entries: {
            name: payload
            for name, payload in entries.items()
            if name != bm.ENCODE_NAME
        },
    )
    with pytest.raises(BundleFormatError):
        DeploymentBundle.load(out)


def test_rejects_oversized_decompressed_bundle(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle whose entries decompress past the cap is refused unread.

    The cap is checked against each entry's declared size before any entry
    is decompressed, so a normal small bundle is enough to trigger it once
    the cap itself is lowered below the bundle's real total -- no need to
    construct an actual multi-hundred-megabyte payload to prove the guard
    works.
    """
    monkeypatch.setattr(bm, "MAX_DECOMPRESSED_BYTES", 8)
    out = _written(tmp_path)
    with pytest.raises(BundleFormatError, match="decompressed size"):
        DeploymentBundle.load(out)


def test_rejects_non_zip_payload(tmp_path: Any) -> None:
    """A file that is not a zip archive is a format error."""
    path = tmp_path / "broken.spkf"
    path.write_bytes(b"not a zip archive")
    with pytest.raises(BundleFormatError):
        DeploymentBundle.load(str(path))


def test_missing_bundle_is_not_found(tmp_path: Any) -> None:
    """A path with no bundle raises the not-found error."""
    with pytest.raises(BundleNotFoundError):
        DeploymentBundle.load(str(tmp_path / "absent.spkf"))


def test_reports_incompatible_torch_version(tmp_path: Any) -> None:
    """A runtime major/minor mismatch is refused under strict loading."""
    out = _written(tmp_path)
    _rewrite(out, lambda entries: _with_torch(entries, "0.0.1"))
    with pytest.raises(BundleCompatibilityError):
        DeploymentBundle.load(out)


def test_strict_can_be_relaxed_for_compatibility(tmp_path: Any) -> None:
    """``strict=False`` reads the bundle despite a version mismatch."""
    out = _written(tmp_path)
    _rewrite(out, lambda entries: _with_torch(entries, "0.0.1"))
    loaded = DeploymentBundle.load(out, strict=False)
    assert loaded.manifest["library_versions"]["torch"] == "0.0.1"


def test_encode_spec_defaults_for_an_empty_bundle() -> None:
    """A directly-built bundle with no config answers with valid defaults."""
    spec, module = registry.build_topology(
        "fc_small", {"hidden": 5, "num_classes": 3}
    )
    bundle = DeploymentBundle(
        manifest={
            "format": bm.BUNDLE_FORMAT,
            "version": bm.BUNDLE_VERSION,
            "spec": spec.to_dict(),
        },
        weights=module.state_dict(),
    )
    assert bundle.has_encode() is False
    resolved = bundle.encode_spec()
    assert resolved.coding == "rate"
    assert resolved.spec_version == bm.ENCODE_SPEC_VERSION
    assert bundle.preprocess_spec().spec_version == bm.ENCODE_SPEC_VERSION


def test_rejects_an_unsupported_encode_spec_version(tmp_path: Any) -> None:
    """A bundle pinning another encode contract is refused under ``strict``."""
    out = _written(tmp_path)
    _rewrite(
        out,
        lambda entries: _with_manifest(entries, encode_spec_version=999),
    )
    with pytest.raises(BundleCompatibilityError):
        DeploymentBundle.load(out)
    relaxed = DeploymentBundle.load(out, strict=False)
    assert relaxed.manifest["encode_spec_version"] == 999
