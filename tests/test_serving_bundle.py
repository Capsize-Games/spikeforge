"""The portable deployment bundle: build, verify, and load."""

import json
import os
import pathlib
import subprocess
import sys
import zipfile
from typing import Any, Dict, Mapping

import pytest
import torch

from spikeforge.network import model_store
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle, build
from spikeforge.serving.errors import (
    BundleCompatibilityError,
    BundleFormatError,
    BundleIntegrityError,
    BundleNotFoundError,
)
from spikeforge.topology import registry
from spikeforge.training.training_engine import TrainingEngine

_NAME = "serving_ckpt"


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint reads and writes into a per-test directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _checkpoint() -> str:
    """Save a small fc_small checkpoint and return its name."""
    torch.manual_seed(0)
    engine = TrainingEngine(
        dataset="mnist",
        num_steps=4,
        device="cpu",
        topology="fc_small",
        topology_params={"hidden": 5, "num_classes": 3},
    )
    engine.save(_NAME)
    return _NAME


def _written(tmp_path: Any) -> str:
    """Build and write a bundle, returning its path."""
    out = str(tmp_path / "model.spkf")
    build(_checkpoint(), out=out)
    return out


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


def test_round_trip_rebuilds_module_and_weights(tmp_path: Any) -> None:
    """A written bundle reloads into the same weights and module."""
    out = str(tmp_path / "model.spkf")
    built = build(_checkpoint(), out=out)
    loaded = DeploymentBundle.load(out)
    assert set(loaded.weights) == set(built.weights)
    for key, value in built.weights.items():
        assert torch.equal(value, loaded.weights[key])
    assert set(loaded.build_module().state_dict()) == set(loaded.weights)


def test_manifest_records_spec_versions_and_entries(tmp_path: Any) -> None:
    """The manifest is self-describing and the entry set is complete."""
    loaded = DeploymentBundle.load(_written(tmp_path))
    manifest = loaded.manifest
    assert manifest["topology"] == "fc_small"
    stages = {s["name"]: s for s in manifest["spec"]["stages"]}
    assert stages["fc2"]["params"]["out_features"] == 3
    assert manifest["spec"]["input"] == "flatten"
    assert manifest["library_versions"]["torch"]
    assert manifest["encode_spec_version"] == bm.ENCODE_SPEC_VERSION
    assert loaded.encode_config["spec_version"] == bm.ENCODE_SPEC_VERSION
    assert loaded.encode_config["coding"] == "rate"
    assert loaded.has_encode() is True
    assert loaded.preprocessing == {}
    with zipfile.ZipFile(loaded.path or "") as archive:
        assert set(bm.REQUIRED_ENTRIES) <= set(archive.namelist())


def test_rebuild_is_bit_exact_in_a_subprocess(tmp_path: Any) -> None:
    """A fresh process rebuilds the module bit-for-bit from the bundle."""
    out = _written(tmp_path)
    built = build(_NAME)
    reference = tmp_path / "reference.pt"
    torch.save(dict(built.weights), reference)
    lines = [
        "import sys, torch",
        "from spikeforge.serving import DeploymentBundle",
        "bundle = DeploymentBundle.load(sys.argv[1])",
        "module = bundle.build_module()",
        "wanted = torch.load(sys.argv[2], map_location='cpu',",
        "                    weights_only=True)",
        "for key, value in module.state_dict().items():",
        "    assert torch.equal(value, wanted[key]), key",
        "print('ok')",
    ]
    root = str(pathlib.Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", "\n".join(lines), out, str(reference)],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=root),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


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
