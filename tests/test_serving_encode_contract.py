"""Issue #3 acceptance: the frozen encode-at-inference contract.

Every check here is the contract itself rather than an implementation detail:
train-time and serve-time encoding must be byte-identical, the bundle must
carry a versioned encode spec, and a version mismatch must be refused instead
of silently re-encoded.
"""

import json
import zipfile
from typing import Any, Dict, Optional

import pytest
import torch

from server.schemas import EncodeConfig
from spikeforge.network import model_store
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving import preprocess
from spikeforge.serving.bundle import DeploymentBundle, build
from spikeforge.serving.encode_spec import (
    ENCODE_SPEC_VERSION,
    EncodeSpec,
)
from spikeforge.serving.errors import BundleCompatibilityError
from spikeforge.serving.session import InferenceSession
from spikeforge.topology import registry
from spikeforge.training.training_engine import TrainingEngine

_NAME = "encode_contract_ckpt"
_RECURRENT = {"hidden": 6, "beta": 0.9, "num_classes": 4}


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint reads and writes into a per-test directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _latency_engine(
    topology: str = "fc_small",
    params: Optional[Dict[str, Any]] = None,
    num_steps: int = 5,
) -> TrainingEngine:
    """Return an engine encoding a deterministic latency code."""
    encode = EncodeConfig(coding="latency", num_steps=num_steps)
    return TrainingEngine(
        dataset="mnist",
        num_steps=num_steps,
        device="cpu",
        topology=topology,
        topology_params=params or {"hidden": 5, "num_classes": 3},
        encode=encode,
    )


def _in_memory_bundle(
    spec: Any, module: Any, encode_config: Dict[str, Any]
) -> DeploymentBundle:
    """Build an in-memory bundle around a live module."""
    return DeploymentBundle(
        manifest={
            "format": bm.BUNDLE_FORMAT,
            "version": bm.BUNDLE_VERSION,
            "spec": spec.to_dict(),
        },
        weights=module.state_dict(),
        encode_config=encode_config,
    )


def _rewrite_manifest_version(path: str, version: int) -> None:
    """Rewrite a bundle's ``encode_spec_version`` and re-sign it."""
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries[bm.MANIFEST_NAME].decode("utf-8"))
    manifest["encode_spec_version"] = version
    entries[bm.MANIFEST_NAME] = bm.dump_json(manifest)
    payloads = {
        name: payload
        for name, payload in entries.items()
        if name != bm.SUMS_NAME
    }
    entries[bm.SUMS_NAME] = bm.sums_text(payloads).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


# --- acceptance 1: byte-identical train and serve encoding --------------


def test_train_and_serve_encoding_are_byte_identical() -> None:
    """A deterministic code is bit-identical whether train or serve runs it."""
    engine = _latency_engine()
    engine.save(_NAME)
    images = torch.rand(2, 1, 28, 28)
    trained = engine._encode_batch(images)
    loaded = build(_NAME)
    spec = loaded.encode_spec()
    served = preprocess.encode(
        images, spec, loaded.spec, geometry=spec.input_size
    )
    assert torch.equal(trained, served)


def test_conv_train_and_serve_encoding_are_byte_identical() -> None:
    """Spatial topologies agree too, geometry included."""
    engine = _latency_engine(
        topology="conv_net",
        params={"channels": 2, "num_classes": 4},
        num_steps=4,
    )
    engine.save(_NAME)
    images = torch.rand(2, 1, 28, 28)
    trained = engine._encode_batch(images)
    loaded = build(_NAME)
    spec = loaded.encode_spec()
    served = preprocess.encode(
        images, spec, loaded.spec, geometry=spec.input_size
    )
    assert torch.equal(trained, served)
    assert served.shape == (4, 2, 1, 28, 28)


# --- acceptance 2: the bundle freezes the spec and pins its version ------


def test_bundle_carries_a_frozen_versioned_encode_spec(tmp_path: Any) -> None:
    """``build`` writes the normalised spec and its contract version."""
    _latency_engine().save(_NAME)
    out = str(tmp_path / "model.spkf")
    build(_NAME, out=out)
    loaded = DeploymentBundle.load(out)
    assert loaded.has_encode()
    assert loaded.encode_config["spec_version"] == ENCODE_SPEC_VERSION
    assert loaded.encode_config["coding"] == "latency"
    assert loaded.manifest["encode_spec_version"] == ENCODE_SPEC_VERSION
    assert loaded.encode_spec().coding == "latency"


def test_bundle_refuses_an_unsupported_encode_spec_version(
    tmp_path: Any,
) -> None:
    """A version mismatch is a compatibility error, not a silent re-encode."""
    _latency_engine().save(_NAME)
    out = str(tmp_path / "model.spkf")
    build(_NAME, out=out)
    _rewrite_manifest_version(out, ENCODE_SPEC_VERSION + 998)
    with pytest.raises(BundleCompatibilityError):
        DeploymentBundle.load(out)
    relaxed = DeploymentBundle.load(out, strict=False)
    assert relaxed.manifest["encode_spec_version"] == ENCODE_SPEC_VERSION + 998


def test_build_defaults_encode_from_checkpoint_meta() -> None:
    """With no encode config, the checkpoint's own meta still freezes one."""
    engine = TrainingEngine(
        dataset="mnist",
        num_steps=4,
        device="cpu",
        topology="fc_small",
        topology_params={"hidden": 5, "num_classes": 3},
    )
    engine.save(_NAME)
    bundle = build(_NAME)
    assert bundle.encode_config["spec_version"] == ENCODE_SPEC_VERSION
    assert bundle.encode_config["coding"] == "rate"


def test_build_prefers_an_explicit_encode_config() -> None:
    """An explicit config overrides the checkpoint default.

    The resulting encode config is normalised.
    """
    TrainingEngine(
        dataset="mnist",
        num_steps=4,
        device="cpu",
        topology="fc_small",
        topology_params={"hidden": 5, "num_classes": 3},
    ).save(_NAME)
    bundle = build(_NAME, encode_config={"coding": "delta", "num_steps": 3})
    assert bundle.encode_config["coding"] == "delta"
    assert bundle.encode_config["num_steps"] == 3
    assert bundle.encode_config["spec_version"] == ENCODE_SPEC_VERSION


# --- acceptance: the session composes encode and step -------------------


def test_step_sample_equals_encode_then_step() -> None:
    """``step_sample`` is exactly ``step(encode(sample))``."""
    spec, module = registry.build_topology("recurrent_net", _RECURRENT)
    encode_config = EncodeSpec(coding="latency", num_steps=5).to_dict()
    bundle = _in_memory_bundle(spec, module, encode_config)
    sample = torch.rand(1, 28, 28)
    composed = InferenceSession.load(bundle).step_sample(sample)
    fresh = InferenceSession.load(bundle)
    manual = fresh.step(fresh.encode(sample))
    assert composed.steps == manual.steps == 1
    assert torch.equal(composed.logits, manual.logits)


def test_run_samples_yields_one_prediction_per_sample() -> None:
    """A raw sample stream advances the session once per sample."""
    spec, module = registry.build_topology("recurrent_net", _RECURRENT)
    encode_config = EncodeSpec(coding="latency", num_steps=5).to_dict()
    bundle = _in_memory_bundle(spec, module, encode_config)
    session = InferenceSession.load(bundle)
    samples = [torch.rand(1, 28, 28) for _ in range(3)]
    predictions = list(session.run_samples(samples))
    assert [prediction.steps for prediction in predictions] == [1, 2, 3]
