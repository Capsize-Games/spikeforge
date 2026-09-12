"""UC-1 P4 acceptance: bundle export and streaming-vs-batch parity.

The served artifact must reproduce the closed-loop reference exactly: the
stateful :class:`InferenceSession` drives the same shared per-step body the
whole-tensor ``run`` uses, and ``spikeforge-serve`` drives the same session.
"""

from typing import Tuple

import pytest
import torch

from spikeforge.network import model_store
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.streaming import (
    StreamSplits,
    StreamTrainConfig,
    StreamTrainingResult,
    build_bundle,
    build_datasets,
    encode_from_bundle,
    encode_windows,
    load_session,
    parity_report,
    predict_batch,
    save_checkpoint,
    stream_logits,
    stream_source,
    train_classifier,
    window_stream,
)
from spikeforge.streaming.serving import PARITY_ATOL, serve_window
from spikeforge_serve.service import ServingService

_SPEC = stream_source.StreamSpec(
    length=8,
    stride=4,
    channels=2,
    classes=2,
    segment_length=16,
    segments=8,
    anomaly_rate=0.4,
    seed=1,
)
_CONFIG = StreamTrainConfig(
    seq_length=8,
    features=2,
    hidden=16,
    num_classes=2,
    num_steps=4,
    epochs=2,
    lr=5e-2,
    seed=0,
)
_NAME = "uc1_parity_model"


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint reads and writes into a per-test directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _trained() -> Tuple[StreamTrainingResult, StreamSplits]:
    """Return a freshly trained result and its splits."""
    splits = build_datasets(_SPEC)
    result = train_classifier(splits.train, _CONFIG)
    save_checkpoint(
        _NAME,
        result,
        splits.window_spec,
        label_map={0: "normal", 1: "fault"},
        metrics={"accuracy": 0.75},
    )
    return result, splits


def _built(tmp_path: object) -> Tuple[StreamTrainingResult, StreamSplits, str]:
    """Train, save, build a ``.spkf``, and return it with its path."""
    result, splits = _trained()
    out = str(tmp_path / "model.spkf")
    build_bundle(_NAME, out=out)
    return result, splits, out


def test_bundle_freezes_encode_and_window_contract(tmp_path: object) -> None:
    """The ``.spkf`` pins the encode spec and the windowing contract."""
    _result, _splits, out = _built(tmp_path)
    bundle = DeploymentBundle.load(out)
    assert bundle.has_encode()
    assert bundle.encode_config["coding"] == _CONFIG.coding
    assert bundle.encode_config["num_steps"] == _CONFIG.num_steps
    assert (
        bundle.manifest["encode_spec_version"]
        == bundle.encode_config["spec_version"]
    )
    window = bundle.preprocessing["window"]
    assert window["length"] == _SPEC.length
    assert window["channels"] == _SPEC.channels
    assert bundle.manifest["label_map"] == {"0": "normal", "1": "fault"}


def test_bundle_reload_rebuilds_the_same_weights(tmp_path: object) -> None:
    """A fresh load rebuilds a module with identical parameters."""
    result, _splits, out = _built(tmp_path)
    bundle = DeploymentBundle.load(out)
    rebuilt = bundle.build_module()
    for key, value in result.module.state_dict().items():
        assert torch.allclose(rebuilt.state_dict()[key], value)


def test_streaming_matches_the_batch_reference(tmp_path: object) -> None:
    """Streaming per-window logits equal the closed-loop batch logits."""
    result, splits, out = _built(tmp_path)
    session = load_session(out)
    windows = splits.test.windows[:8]
    report = parity_report(result.module, session, windows, result.encode_spec)
    assert report["max_abs_diff"] <= PARITY_ATOL
    assert report["within_tolerance"] is True
    assert report["batch_labels"] == report["stream_labels"]


def test_streaming_matches_batch_for_a_single_window(tmp_path: object) -> None:
    """The parity holds for one window, the smallest streaming unit."""
    result, splits, out = _built(tmp_path)
    session = load_session(out)
    window = splits.test.windows[:1]
    batch = predict_batch(result.module, window, result.encode_spec)
    streamed = stream_logits(session, window, result.encode_spec)
    assert torch.allclose(batch, streamed, atol=PARITY_ATOL)


def test_session_reset_clears_carried_state(tmp_path: object) -> None:
    """``reset`` returns the session to an unstepped, stateless state."""
    result, splits, out = _built(tmp_path)
    session = load_session(out)
    spikes = encode_windows(splits.test.windows[:1], result.encode_spec)
    for step in range(int(spikes.size(0))):
        session.step(spikes[step])
    assert session.steps == _CONFIG.num_steps
    session.reset()
    assert session.steps == 0
    assert session.state()["state"] == {}


def test_served_window_matches_the_batch_reference(tmp_path: object) -> None:
    """The ``spikeforge-serve`` service returns the same readout as batch."""
    result, splits, out = _built(tmp_path)
    service = ServingService(out)
    window = splits.test.windows[:1]
    served = serve_window(service, window, result.encode_spec)
    batch = predict_batch(result.module, window, result.encode_spec)
    assert torch.allclose(
        served.mean_logits.reshape(1, -1), batch, atol=PARITY_ATOL
    )
    assert service.reset() == 0


def test_encode_from_bundle_matches_the_training_windows(
    tmp_path: object,
) -> None:
    """A raw window encoded from the bundle equals the train-path encoding."""
    result, splits, out = _built(tmp_path)
    bundle = DeploymentBundle.load(out)
    raw_stream, _labels, _anomaly = stream_source.generate_stream(
        _SPEC, _SPEC.seed + stream_source._TEST_SEED_OFFSET
    )
    raw = window_stream(raw_stream, _SPEC.length, _SPEC.stride)
    from_bundle = encode_from_bundle(bundle, raw)
    from_training = encode_windows(splits.test.windows, result.encode_spec)
    assert from_bundle.shape == from_training.shape
    assert torch.allclose(from_bundle, from_training, atol=1e-6)
