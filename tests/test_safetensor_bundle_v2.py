"""Version 2 SafeTensors bundles: build, verify, import, activate."""

import json
import zipfile
from typing import Any, Callable, Dict

import pytest
import torch

pytest.importorskip("safetensors", reason="serve extra not installed")

from spikeforge.network import model_store
from spikeforge.serving import model_activation as act
from spikeforge.serving import runtime_map as rt
from spikeforge.serving.bundle import build as build_v1
from spikeforge.serving.errors import BundleFormatError, BundleIntegrityError
from spikeforge.serving.safetensor_bundle import STBundle
from spikeforge.serving.st_build import build_st_bundle
from spikeforge.serving.st_format import (
    MANIFEST_NAME,
    MODEL_NAME,
    SUMS_NAME,
    WEIGHTS_NAME,
    checksum,
)
from spikeforge.serving.st_import import import_v1_bundle
from spikeforge.training.training_engine import TrainingEngine


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: Any) -> None:
    """Redirect checkpoint reads and writes into a test directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _checkpoint(hidden: int = 5) -> str:
    """Save a small fc_small checkpoint and return its name."""
    torch.manual_seed(0)
    engine = TrainingEngine(
        dataset="mnist", num_steps=4, device="cpu",
        topology="fc_small",
        topology_params={"hidden": hidden, "num_classes": 3},
    )
    engine.save(f"st_ckpt_{hidden}")
    return f"st_ckpt_{hidden}"


def _built(tmp_path: Any, name: str = "m.stbk", **kw: Any) -> str:
    """Build and write a version 2 bundle, returning its path."""
    out = str(tmp_path / name)
    build_st_bundle(_checkpoint(int(kw.pop("hidden", 5))), out=out, **kw)
    return out


def _entries(path: str) -> Dict[str, bytes]:
    """Return the raw entries of the bundle at ``path``."""
    with zipfile.ZipFile(path) as archive:
        return {n: archive.read(n) for n in archive.namelist()}


def _rewrite(path: str, mutate: Callable, regen: bool = True) -> None:
    """Rewrite a bundle, optionally recomputing its SHA256SUMS."""
    entries = mutate(_entries(path))
    if regen:
        payloads = {n: p for n, p in entries.items() if n != SUMS_NAME}
        body = "".join(f"{checksum(p)}  {n}\n" for n, p in payloads.items())
        entries[SUMS_NAME] = body.encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def _with_manifest(
    entries: Dict[str, bytes], **changes: Any
) -> Dict[str, bytes]:
    """Return ``entries`` with top-level manifest keys updated."""
    manifest = json.loads(entries[MANIFEST_NAME].decode("utf-8"))
    manifest.update(changes)
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode()
    return {**entries, MANIFEST_NAME: payload}


def test_roundtrip_preserves_bytes_and_outputs(tmp_path: Any) -> None:
    """Build, reload, and re-run: weights and outputs agree exactly."""
    built = build_st_bundle(_checkpoint(), expected={"probe": [0.1]})
    out = str(tmp_path / "m.stbk")
    built.save(out)
    loaded = STBundle.load(out)
    assert set(loaded.weights) == set(built.weights)
    for key, value in built.weights.items():
        assert str(loaded.weights[key].dtype) == str(value.dtype)
        assert list(loaded.weights[key].shape) == list(value.shape)
        assert torch.equal(loaded.weights[key], value)
    _assert_parity(built, loaded)


def _assert_parity(built: STBundle, loaded: STBundle) -> None:
    """Assert both bundles run one identical inference step."""
    probe = torch.randn(1, 784)
    first, _ = built.build_module().step(probe)
    second, _ = loaded.build_module().step(probe)
    assert set(first) == set(second)
    for key in first:
        assert torch.equal(first[key], second[key])


def test_rejects_tampered_weights(tmp_path: Any) -> None:
    """A byte changed after signing is an integrity error."""
    out = _built(tmp_path)

    def _flip(entries: Dict[str, bytes]) -> Dict[str, bytes]:
        raw = bytearray(entries[WEIGHTS_NAME])
        raw[100] = (raw[100] + 1) % 256
        return {**entries, WEIGHTS_NAME: bytes(raw)}

    _rewrite(out, _flip, regen=False)
    with pytest.raises(BundleIntegrityError) as info:
        STBundle.load(out)
    assert info.value.entry == WEIGHTS_NAME


def test_rejects_bad_shape(tmp_path: Any) -> None:
    """A manifest shape disagreeing with weights is refused."""
    out = _built(tmp_path)
    manifest = json.loads(_entries(out)[MANIFEST_NAME].decode("utf-8"))
    first = sorted(manifest["tensors"])[0]
    manifest["tensors"][first]["shape"] = [1]
    _rewrite(out, lambda e: _with_manifest(e, tensors=manifest["tensors"]))
    with pytest.raises(BundleFormatError, match="shape"):
        STBundle.load(out)


def test_rejects_mixed_versions(tmp_path: Any) -> None:
    """Version 1 bytes never load as version 2; versions must match."""
    out = _built(tmp_path)
    _rewrite(out, lambda e: _with_manifest(e, version=1))
    with pytest.raises(BundleFormatError, match="version"):
        STBundle.load(out)
    legacy = str(tmp_path / "legacy.spkf")
    build_v1(_checkpoint(), out=legacy)
    with pytest.raises(BundleFormatError):
        STBundle.load(legacy)


def test_rejects_inconsistent_metadata(tmp_path: Any) -> None:
    """Neuron metadata disagreeing with the spec is refused."""
    out = _built(tmp_path)

    def _drop(entries: Dict[str, bytes]) -> Dict[str, bytes]:
        doc = json.loads(entries[MODEL_NAME].decode("utf-8"))
        doc["stage_neurons"].pop(sorted(doc["stage_neurons"])[0])
        raw = json.dumps(doc, indent=2, sort_keys=True).encode()
        manifest = json.loads(entries[MANIFEST_NAME].decode("utf-8"))
        manifest["model_digest"] = checksum(raw)
        dumped = json.dumps(manifest, indent=2, sort_keys=True).encode()
        return {**entries, MODEL_NAME: raw, MANIFEST_NAME: dumped}

    _rewrite(out, _drop)
    with pytest.raises(BundleFormatError, match="neuron"):
        STBundle.load(out)


def test_rejects_wrong_hash_for_session(tmp_path: Any) -> None:
    """A checkpoint bound to another model's hash is refused."""
    bundle = STBundle.load(_built(tmp_path))
    other = STBundle.load(_built(tmp_path, name="b.stbk", hidden=7))
    ref = dict(act.model_ref(bundle))
    act.check_model_ref(ref, bundle)
    with pytest.raises(BundleIntegrityError):
        act.check_model_ref(dict(ref, model_hash=other.model_hash), bundle)
    with pytest.raises(BundleFormatError):
        act.check_model_ref({"schema": "other",
                             "model_hash": ref["model_hash"]}, bundle)


def test_v1_import_preserves_and_classifies(tmp_path: Any) -> None:
    """v1 converts byte-exact; gaps are listed, never invented."""
    legacy = tmp_path / "legacy.spkf"
    build_v1(_checkpoint(), out=str(legacy))
    before = checksum(legacy.read_bytes())
    converted, report = import_v1_bundle(str(legacy), expected={"q": [0.5]})
    assert "expected_outputs" not in report["missing"]
    assert "provenance.dataset" in report["missing"]
    assert converted.manifest["expected_outputs"] == {"q": [0.5]}
    assert converted.manifest["provenance"]["source_sha256"] == before
    assert "weights.pt" in converted.manifest["provenance"]["source_entries"]
    assert checksum(legacy.read_bytes()) == before
    reloaded = STBundle.load(converted.save(str(tmp_path / "c.stbk")))
    assert set(reloaded.weights) == set(converted.weights)


def test_activation_rollback_and_restart(tmp_path: Any) -> None:
    """Failed updates keep the old version; restarts reload one good."""
    root = str(tmp_path / "cache")
    cache = act.ActivationCache(root)
    first = _built(tmp_path, name="a.stbk")
    second = _built(tmp_path, name="b.stbk", hidden=7)
    hash_a = cache.activate(first)
    junk = str(tmp_path / "junk")
    (tmp_path / "junk").write_bytes(b"junk")
    with pytest.raises(BundleFormatError):
        cache.activate(junk)
    assert cache.active_hash() == hash_a
    (tmp_path / "cache" / "active.json.tmp").write_bytes(b"partial")
    assert act.ActivationCache(root).active_hash() == hash_a
    cache.activate(second)
    assert cache.rollback() == hash_a
    assert act.ActivationCache(root).active_hash() == hash_a


def _churn(module: Any, state: Any) -> None:
    """Advance ``module`` three steps, dropping every output."""
    current = state
    for _ in range(3):
        _, current = module.step(torch.zeros(1, 784), current)


def test_workers_share_weights_not_state(tmp_path: Any) -> None:
    """One artifact feeds two modules with isolated temporal states."""
    path = _built(tmp_path)
    bundle = STBundle.load(path)
    mod_a, mod_b = bundle.build_module(), bundle.build_module()
    probe = torch.ones(1, 784)
    out_a, state_a = mod_a.step(probe)
    out_b, state_b = mod_b.step(probe)
    for key in out_a:
        assert torch.equal(out_a[key], out_b[key])
    assert state_a is not state_b
    _churn(mod_b, state_b)
    rival = STBundle.load(path).build_module()
    _, rival_state = rival.step(probe)
    want, _ = rival.step(probe, rival_state)
    got, _ = mod_a.step(probe, state_a)
    assert torch.equal(want["lif2"], got["lif2"])
    clones = act.fresh_weights(bundle)
    assert set(clones) == set(bundle.weights)


def test_runtime_map_enumerates_supported_nodes(tmp_path: Any) -> None:
    """The native summary lists real nodes; all are supported kinds."""
    bundle = STBundle.load(_built(tmp_path))
    summary = rt.map_to_runtime_summary(bundle)
    report = rt.capability_report(summary)
    assert report["node_count"] > 0
    assert report["edge_count"] > 0
    assert report["unsupported"] == []
    assert report["binary_capability"]["version"] == 1
