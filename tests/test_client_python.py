"""PT-W4 acceptance: the Python ``spikeforge-clients`` client.

The client is exercised through the dependency-free in-process ASGI transport
against the real ``spikeforge-serve`` app, so ``predict`` and ``stream`` are
proven to round-trip against the server rather than a stub: the reference is
the core :class:`~spikeforge.serving.session.InferenceSession` the server
itself uses.
"""

from pathlib import Path
from typing import Any, List

import pytest
import torch

from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving import preprocess
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.encode_spec import ENCODE_SPEC_VERSION, EncodeSpec
from spikeforge.serving.session import InferenceSession
from spikeforge.topology import registry
from spikeforge_clients import ServeClient, ServiceError, TransportError
from spikeforge_clients.errors import describe
from spikeforge_clients.prediction import PredictionResult
from spikeforge_serve import create_app

_TOPOLOGY = "fc_small"
_PARAMS = {"hidden": 5, "num_classes": 3}
_NUM_STEPS = 4


def _write_bundle(path: Path) -> None:
    """Write a small, deterministic ``.spkf`` bundle for the tests."""
    spec, module = registry.build_topology(_TOPOLOGY, dict(_PARAMS))
    module.eval()
    bundle = DeploymentBundle(
        manifest={
            "format": bm.BUNDLE_FORMAT,
            "version": bm.BUNDLE_VERSION,
            "spec": spec.to_dict(),
            "topology": _TOPOLOGY,
            "topology_params": dict(_PARAMS),
            "encode_spec_version": ENCODE_SPEC_VERSION,
            "num_classes": _PARAMS["num_classes"],
            "label_map": {"0": "a", "1": "b", "2": "c"},
            "expected_metrics": {"test_accuracy": 0.9},
        },
        weights=module.state_dict(),
        encode_config=EncodeSpec(
            coding="latency", num_steps=_NUM_STEPS
        ).to_dict(),
    )
    bundle.save(str(path))


def _sample(seed: int) -> List[Any]:
    """Return one deterministic ``[1, 28, 28]`` raw sample as JSON."""
    torch.manual_seed(seed)
    return torch.rand(1, 28, 28).tolist()


@pytest.fixture()
def bundle_path(tmp_path: Path) -> str:
    """Return the path to a freshly written test bundle."""
    path = tmp_path / "model.spkf"
    _write_bundle(path)
    return str(path)


@pytest.fixture()
def client(bundle_path: str) -> ServeClient:
    """Return a client driving the in-process serve app."""
    return ServeClient.in_process(create_app(bundle_path))


# --- probes and bundle metadata ------------------------------------------


def test_health_and_ready_report_success(client: ServeClient) -> None:
    """``health`` and ``ready`` decode into their typed results."""
    assert client.health().status == "ok"
    ready = client.ready()
    assert ready.ready is True
    assert ready.status == "ready"


def test_bundle_info_is_typed(client: ServeClient) -> None:
    """``bundle_info`` exposes the fields a client shapes its input with."""
    info = client.bundle_info()
    assert info.format == bm.BUNDLE_FORMAT
    assert info.topology == _TOPOLOGY
    assert info.num_classes == _PARAMS["num_classes"]
    assert info.encode_spec["coding"] == "latency"
    assert info.encode_spec["num_steps"] == _NUM_STEPS
    assert info.label_map["0"] == "a"


# --- predict --------------------------------------------------------------


def test_predict_round_trips_against_the_server(
    client: ServeClient, bundle_path: str
) -> None:
    """``predict`` reproduces the in-process reference for each frame."""
    samples = [_sample(1), _sample(2)]
    session = InferenceSession.load(bundle_path)
    expected = [
        list(session.run_stream(session.encode(sample)))[-1]
        for sample in samples
    ]
    response = client.predict(samples)
    assert response.session_id
    assert response.steps == _NUM_STEPS * len(samples)
    assert len(response.predictions) == len(expected)
    for actual, reference in zip(response.predictions, expected):
        assert isinstance(actual, PredictionResult)
        assert actual.label == reference.label
        assert actual.steps == reference.steps
        assert actual.logits.shape == tuple(reference.logits.shape)
        assert actual.logits.values[0] == pytest.approx(
            reference.logits.tolist()[0]
        )


def test_predict_carries_state_and_isolates_sessions(
    client: ServeClient,
) -> None:
    """Sequential calls accumulate; a named session keeps its own count."""
    sample = _sample(3)
    first = client.predict([sample])
    second = client.predict([sample])
    assert first.steps == _NUM_STEPS
    assert second.steps == 2 * _NUM_STEPS
    other = client.predict([sample], session_id="b")
    assert other.session_id == "b"
    assert other.steps == _NUM_STEPS


def test_predict_accepts_pre_encoded_frames(
    client: ServeClient, bundle_path: str
) -> None:
    """A batch of pre-encoded frames is stepped without re-encoding."""
    bundle = DeploymentBundle.load(bundle_path)
    spec = bundle.encode_spec()
    spikes = preprocess.encode(
        _sample(6), spec, bundle.spec, geometry=spec.input_size
    )
    frames = [row.tolist() for row in spikes]
    response = client.predict(frames, encoded=True)
    assert response.steps == _NUM_STEPS
    assert isinstance(response.predictions[-1], PredictionResult)


def test_reset_clears_temporal_state(client: ServeClient) -> None:
    """``reset`` zeroes the step count and restarts the stream."""
    sample = _sample(5)
    client.predict([sample])
    result = client.reset()
    assert result.steps == 0
    assert client.predict([sample]).steps == _NUM_STEPS


def test_reset_reports_a_named_session(client: ServeClient) -> None:
    """``reset`` echoes the session id it cleared."""
    result = client.reset("named")
    assert result.session_id == "named"
    assert result.steps == 0


# --- stream ---------------------------------------------------------------


def test_stream_predicts_like_the_server(
    client: ServeClient, bundle_path: str
) -> None:
    """``stream`` yields one prediction per frame with carried state."""
    sample = _sample(7)
    events = list(client.stream([sample, sample, sample], session_id="s"))
    assert [event.type for event in events] == ["prediction"] * 3
    assert [event.steps for event in events] == [
        _NUM_STEPS,
        2 * _NUM_STEPS,
        3 * _NUM_STEPS,
    ]
    session = InferenceSession.load(bundle_path)
    reference = [
        list(session.run_stream(session.encode(sample)))[-1]
        for _ in range(3)
    ]
    assert [event.label for event in events] == [
        item.label for item in reference
    ]
    prediction = events[0].prediction()
    assert prediction is not None
    assert prediction.label == reference[0].label


# --- error mapping --------------------------------------------------------


def test_missing_bundle_maps_to_a_typed_error(tmp_path: Path) -> None:
    """A missing bundle is a typed 404 and leaves the service unready."""
    client = ServeClient.in_process(
        create_app(str(tmp_path / "absent.spkf"))
    )
    assert client.health().status == "ok"
    unready = client.ready()
    assert unready.ready is False
    assert unready.status == "bundle_not_found"
    with pytest.raises(ServiceError) as error:
        client.bundle_info()
    assert error.value.status == 404
    assert error.value.error_type == "bundle_not_found"


def test_bad_predict_body_maps_to_a_typed_error(
    client: ServeClient,
) -> None:
    """An empty frame list is a typed 400 with the service's message."""
    with pytest.raises(ServiceError) as error:
        client.predict([])
    assert error.value.status == 400
    assert error.value.error_type == "http_error"


def test_transport_error_names_the_failure() -> None:
    """A refused connection is a typed ``TransportError``, not a traceback."""
    client = ServeClient(base_url="http://127.0.0.1:1", timeout=0.5)
    with pytest.raises(TransportError):
        client.health()


def test_describe_handles_both_error_shapes() -> None:
    """``describe`` reads the serving taxonomy and the validation shape."""
    assert describe({"error": {"type": "x", "message": "y"}}) == ("x", "y")
    assert describe({"detail": "bad"}) == ("http_error", "bad")
    assert describe(None) == ("http_error", "service returned an error "
                              "status")
