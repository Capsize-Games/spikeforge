"""Issue #4 acceptance: the headless ``spikeforge-serve`` inference service.

The service is exercised through its ASGI interface directly, so the tests
cover the real routes without depending on ``httpx``/``TestClient``: a tiny
in-process driver feeds ASGI http and websocket scopes to the app returned by
``create_app`` and collects the responses.
"""


import asyncio
import json
from pathlib import Path

import pytest
from _fakes import _get_json, _post_json, _websocket
from _serve_helpers import (
    _NUM_STEPS,
    _PARAMS,
    _TOPOLOGY,
    _sample,
    _write_bundle,
)

from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving import preprocess
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.session import InferenceSession
from spikeforge_serve import create_app
from spikeforge_serve.__main__ import _parse_args


@pytest.fixture()
def bundle_path(tmp_path: Path) -> str:
    """Return the path to a freshly written test bundle."""
    path = tmp_path / "model.spkf"
    _write_bundle(path)
    return str(path)


# --- app creation and probes ---------------------------------------------


def test_create_app_returns_an_asgi_app(bundle_path: str) -> None:
    """The factory returns an ASGI app without binding a port."""
    app = create_app(bundle_path)
    assert app.state.serving.bundle.path == bundle_path


def test_health_endpoint_is_always_ok(bundle_path: str) -> None:
    """``/health`` reports liveness even before the bundle is loaded."""
    status, payload = _get_json(create_app(bundle_path), "/health")
    assert status == 200
    assert payload == {"status": "ok"}


def test_ready_endpoint_reports_the_loaded_bundle(bundle_path: str) -> None:
    """``/ready`` loads the bundle and reports readiness."""
    status, payload = _get_json(create_app(bundle_path), "/ready")
    assert status == 200
    assert payload == {"status": "ready"}


def test_bundle_endpoint_reports_metadata(bundle_path: str) -> None:
    """``/v1/bundle`` returns the self-describing bundle metadata."""
    status, payload = _get_json(create_app(bundle_path), "/v1/bundle")
    assert status == 200
    assert payload["format"] == bm.BUNDLE_FORMAT
    assert payload["topology"] == _TOPOLOGY
    assert payload["num_classes"] == _PARAMS["num_classes"]
    assert payload["encode_spec"]["coding"] == "latency"
    assert payload["encode_spec"]["num_steps"] == _NUM_STEPS
    assert payload["label_map"]["0"] == "a"
    assert payload["expected_metrics"]["test_accuracy"] == 0.9
    assert payload["path"] == bundle_path


def test_create_app_accepts_a_loaded_bundle(bundle_path: str) -> None:
    """An already-loaded bundle is served without a second disk read."""
    bundle = DeploymentBundle.load(bundle_path)
    status, payload = _get_json(create_app(bundle), "/v1/bundle")
    assert status == 200
    assert payload["topology"] == _TOPOLOGY


def test_stream_route_describes_the_protocol(bundle_path: str) -> None:
    """``GET /v1/stream`` documents the WebSocket protocol."""
    status, payload = _get_json(create_app(bundle_path), "/v1/stream")
    assert status == 200
    assert payload["protocol"] == "websocket"


# --- inference ------------------------------------------------------------


def test_predict_matches_in_process_reference(bundle_path: str) -> None:
    """``predict`` over raw samples reproduces the in-process reference."""
    samples = [_sample(1), _sample(2)]
    session = InferenceSession.load(bundle_path)
    expected = [
        list(session.run_stream(session.encode(sample)))[-1]
        for sample in samples
    ]
    status, payload = _post_json(
        create_app(bundle_path), "/v1/predict", {"frames": samples}
    )
    assert status == 200
    assert payload["steps"] == _NUM_STEPS * len(samples)
    got = payload["predictions"]
    assert len(got) == len(expected)
    for actual, reference in zip(got, expected):
        assert actual["label"] == reference.label
        assert actual["steps"] == reference.steps
        assert actual["logits"]["shape"] == list(reference.logits.shape)
        assert actual["logits"]["values"][0] == pytest.approx(
            reference.logits.tolist()[0]
        )
        assert actual["class_totals"]["values"][0] == pytest.approx(
            reference.class_totals.tolist()[0]
        )


def test_predict_carries_state_across_requests(bundle_path: str) -> None:
    """Sequential requests accumulate the session's temporal state."""
    app = create_app(bundle_path)
    sample = _sample(3)
    _, first = _post_json(app, "/v1/predict", {"frames": [sample]})
    _, second = _post_json(app, "/v1/predict", {"frames": [sample]})
    assert first["steps"] == _NUM_STEPS
    assert second["steps"] == 2 * _NUM_STEPS


def test_session_ids_are_isolated(bundle_path: str) -> None:
    """A named session keeps its own step count."""
    app = create_app(bundle_path)
    sample = _sample(4)
    _post_json(
        app,
        "/v1/predict",
        {"frames": [sample], "session_id": "a"},
    )
    _, other = _post_json(
        app,
        "/v1/predict",
        {"frames": [sample], "session_id": "b"},
    )
    assert other["steps"] == _NUM_STEPS


def test_reset_clears_temporal_state(bundle_path: str) -> None:
    """``/v1/reset`` drops the carried state and restarts the stream."""
    app = create_app(bundle_path)
    sample = _sample(5)
    _post_json(app, "/v1/predict", {"frames": [sample]})
    status, payload = _post_json(app, "/v1/reset", {"session_id": "default"})
    assert status == 200
    assert payload == {"session_id": "default", "steps": 0}
    _, again = _post_json(app, "/v1/predict", {"frames": [sample]})
    assert again["steps"] == _NUM_STEPS


def test_predict_accepts_pre_encoded_frames(bundle_path: str) -> None:
    """A batch of pre-encoded frames is stepped without re-encoding."""
    bundle = DeploymentBundle.load(bundle_path)
    spec = bundle.encode_spec()
    spikes = preprocess.encode(
        _sample(6), spec, bundle.spec, geometry=spec.input_size
    )
    frames = [row.tolist() for row in spikes]
    session = InferenceSession.load(bundle_path)
    expected = [session.step(row) for row in spikes][-1]
    status, payload = _post_json(
        create_app(bundle_path),
        "/v1/predict",
        {"frames": frames, "encoded": True},
    )
    assert status == 200
    assert payload["steps"] == _NUM_STEPS
    assert payload["predictions"][-1]["label"] == expected.label


def test_stream_websocket_predicts_and_resets(bundle_path: str) -> None:
    """``WS /v1/stream`` steps frames, carries state, and resets on demand."""
    app = create_app(bundle_path)
    sample = _sample(7)
    sent = asyncio.run(
        _websocket(
            app,
            "/v1/stream",
            [{"frame": sample}, {"frame": sample}, {"reset": True}],
        )
    )
    frames = [item for item in sent if item["type"] == "websocket.send"]
    replies = [json.loads(item["text"]) for item in frames]
    assert [reply["type"] for reply in replies] == [
        "prediction",
        "prediction",
        "reset",
    ]
    assert replies[0]["payload"]["steps"] == _NUM_STEPS
    assert replies[1]["payload"]["steps"] == 2 * _NUM_STEPS
    assert replies[1]["payload"]["label"] in range(
        _PARAMS["num_classes"]
    )
    assert replies[2]["payload"]["steps"] == 0


# --- CLI ------------------------------------------------------------------


def test_cli_parses_bundle_and_bind_arguments() -> None:
    """The ``serve`` subcommand accepts the bundle path and bind overrides."""
    args = _parse_args(
        [
            "serve", "--bundle", "model.spkf",
            "--host", "1.2.3.4", "--port", "9000",
        ]
    )
    assert args.command == "serve"
    assert args.bundle == "model.spkf"
    assert args.host == "1.2.3.4"
    assert args.port == 9000
    assert args.device == "cpu"


def test_cli_serve_requires_a_bundle() -> None:
    """Omitting ``--bundle`` from ``serve`` is a usage error."""
    with pytest.raises(SystemExit):
        _parse_args(["serve"])


def test_cli_requires_a_command() -> None:
    """Omitting the subcommand entirely is a usage error."""
    with pytest.raises(SystemExit):
        _parse_args([])
