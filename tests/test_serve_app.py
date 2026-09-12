"""Issue #4 acceptance: the headless ``spikeforge-serve`` inference service.

The service is exercised through its ASGI interface directly, so the tests
cover the real routes without depending on ``httpx``/``TestClient``: a tiny
in-process driver feeds ASGI http and websocket scopes to the app returned by
``create_app`` and collects the responses.
"""

import asyncio
import json
import platform
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch

from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving import preprocess
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.encode_spec import ENCODE_SPEC_VERSION, EncodeSpec
from spikeforge.serving.session import InferenceSession
from spikeforge.topology import registry
from spikeforge_serve import create_app
from spikeforge_serve.__main__ import _parse_args

_TOPOLOGY = "fc_small"
_PARAMS = {"hidden": 5, "num_classes": 3}
_NUM_STEPS = 4


# --- fixtures and helpers ------------------------------------------------


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


def _rewrite_manifest(
    path: Path, mutate: Callable[[Dict[str, Any]], None]
) -> None:
    """Rewrite a bundle's manifest, then re-sign its checksums."""
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries[bm.MANIFEST_NAME].decode("utf-8"))
    mutate(manifest)
    entries[bm.MANIFEST_NAME] = bm.dump_json(manifest)
    payloads = {
        name: value
        for name, value in entries.items()
        if name != bm.SUMS_NAME
    }
    entries[bm.SUMS_NAME] = bm.sums_text(payloads).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)


@pytest.fixture()
def bundle_path(tmp_path: Path) -> str:
    """Return the path to a freshly written test bundle."""
    path = tmp_path / "model.spkf"
    _write_bundle(path)
    return str(path)


# --- a dependency-free ASGI driver ---------------------------------------


def _headers(
    has_body: bool, extra: Optional[Dict[str, str]]
) -> List[Tuple[bytes, bytes]]:
    """Return ASGI header pairs for a JSON request."""
    headers: List[Tuple[bytes, bytes]] = []
    if has_body:
        headers.append((b"content-type", b"application/json"))
    for name, value in (extra or {}).items():
        headers.append((name.lower().encode("utf-8"), value.encode("utf-8")))
    return headers


async def _http(
    app: Any,
    method: str,
    path: str,
    body: Any = None,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Any]:
    """Drive ``app`` as an ASGI HTTP request and return ``(status, json)``."""
    raw = b"" if body is None else json.dumps(body).encode("utf-8")
    sent: List[Dict[str, Any]] = []

    async def receive() -> Dict[str, Any]:
        """Return the whole request body once."""
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(message: Dict[str, Any]) -> None:
        """Record one response message."""
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "root_path": "",
        "headers": _headers(body is not None, headers),
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    status = next(
        item["status"]
        for item in sent
        if item["type"] == "http.response.start"
    )
    chunks = [
        item["body"]
        for item in sent
        if item["type"] == "http.response.body"
    ]
    return status, json.loads(b"".join(chunks) or b"null")


async def _websocket(
    app: Any, path: str, messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Drive ``app`` as an ASGI WebSocket session and return sent frames."""
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put({"type": "websocket.connect"})
    for message in messages:
        await queue.put(
            {"type": "websocket.receive", "text": json.dumps(message)}
        )
    await queue.put({"type": "websocket.disconnect", "code": 1000})
    sent: List[Dict[str, Any]] = []

    async def receive() -> Dict[str, Any]:
        """Return the next queued client message."""
        return await queue.get()

    async def send(message: Dict[str, Any]) -> None:
        """Record one server frame."""
        sent.append(message)

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "subprotocols": [],
    }
    await app(scope, receive, send)
    return sent


def _get(app: Any, path: str) -> Tuple[int, Any]:
    """Issue a GET request against ``app``."""
    return asyncio.run(_http(app, "GET", path))


def _post(app: Any, path: str, body: Any) -> Tuple[int, Any]:
    """Issue a POST request against ``app``."""
    return asyncio.run(_http(app, "POST", path, body))


# --- app creation and probes ---------------------------------------------


def test_create_app_returns_an_asgi_app(bundle_path: str) -> None:
    """The factory returns an ASGI app without binding a port."""
    app = create_app(bundle_path)
    assert app.state.serving.bundle.path == bundle_path


def test_health_endpoint_is_always_ok(bundle_path: str) -> None:
    """``/health`` reports liveness even before the bundle is loaded."""
    status, payload = _get(create_app(bundle_path), "/health")
    assert status == 200
    assert payload == {"status": "ok"}


def test_ready_endpoint_reports_the_loaded_bundle(bundle_path: str) -> None:
    """``/ready`` loads the bundle and reports readiness."""
    status, payload = _get(create_app(bundle_path), "/ready")
    assert status == 200
    assert payload == {"status": "ready"}


def test_bundle_endpoint_reports_metadata(bundle_path: str) -> None:
    """``/v1/bundle`` returns the self-describing bundle metadata."""
    status, payload = _get(create_app(bundle_path), "/v1/bundle")
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
    status, payload = _get(create_app(bundle), "/v1/bundle")
    assert status == 200
    assert payload["topology"] == _TOPOLOGY


def test_stream_route_describes_the_protocol(bundle_path: str) -> None:
    """``GET /v1/stream`` documents the WebSocket protocol."""
    status, payload = _get(create_app(bundle_path), "/v1/stream")
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
    status, payload = _post(
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
    _, first = _post(app, "/v1/predict", {"frames": [sample]})
    _, second = _post(app, "/v1/predict", {"frames": [sample]})
    assert first["steps"] == _NUM_STEPS
    assert second["steps"] == 2 * _NUM_STEPS


def test_session_ids_are_isolated(bundle_path: str) -> None:
    """A named session keeps its own step count."""
    app = create_app(bundle_path)
    sample = _sample(4)
    _post(
        app,
        "/v1/predict",
        {"frames": [sample], "session_id": "a"},
    )
    _, other = _post(
        app,
        "/v1/predict",
        {"frames": [sample], "session_id": "b"},
    )
    assert other["steps"] == _NUM_STEPS


def test_reset_clears_temporal_state(bundle_path: str) -> None:
    """``/v1/reset`` drops the carried state and restarts the stream."""
    app = create_app(bundle_path)
    sample = _sample(5)
    _post(app, "/v1/predict", {"frames": [sample]})
    status, payload = _post(app, "/v1/reset", {"session_id": "default"})
    assert status == 200
    assert payload == {"session_id": "default", "steps": 0}
    _, again = _post(app, "/v1/predict", {"frames": [sample]})
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
    status, payload = _post(
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


# --- error mapping --------------------------------------------------------


def test_missing_bundle_maps_to_404(tmp_path: Path) -> None:
    """A missing bundle is a 404 and leaves the service unready."""
    app = create_app(str(tmp_path / "absent.spkf"))
    status, payload = _get(app, "/v1/bundle")
    assert status == 404
    assert payload["error"]["type"] == "bundle_not_found"
    assert _get(app, "/health")[0] == 200
    assert _get(app, "/ready")[0] == 503


def test_malformed_bundle_maps_to_422(tmp_path: Path) -> None:
    """A file that is not a zip bundle is an unprocessable entity."""
    path = tmp_path / "broken.spkf"
    path.write_bytes(b"not a zip archive")
    status, payload = _get(create_app(str(path)), "/v1/bundle")
    assert status == 422
    assert payload["error"]["type"] == "bundle_format"


def test_tampered_bundle_maps_to_422_integrity(tmp_path: Path) -> None:
    """A bundle whose contents do not match its checksums is refused."""
    path = tmp_path / "tampered.spkf"
    _write_bundle(path)
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries[bm.WEIGHTS_NAME] = entries[bm.WEIGHTS_NAME] + b"tampered"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    status, payload = _get(create_app(str(path)), "/v1/bundle")
    assert status == 422
    assert payload["error"]["type"] == "bundle_integrity"


def test_incompatible_bundle_maps_to_409(tmp_path: Path) -> None:
    """A bundle built by an incompatible runtime is a conflict."""
    path = tmp_path / "incompatible.spkf"
    _write_bundle(path)

    def _downgrade(manifest: Dict[str, Any]) -> None:
        """Record an impossible torch version against the manifest."""
        manifest["library_versions"] = {
            "torch": "0.0",
            "python": platform.python_version(),
        }

    _rewrite_manifest(path, _downgrade)
    status, payload = _get(create_app(str(path)), "/v1/bundle")
    assert status == 409
    assert payload["error"]["type"] == "bundle_compatibility"


def test_predict_rejects_a_bad_body(bundle_path: str) -> None:
    """A request without a non-empty frame list is a 400."""
    app = create_app(bundle_path)
    assert _post(app, "/v1/predict", {})[0] == 400
    assert _post(app, "/v1/predict", {"frames": []})[0] == 400
    assert _post(app, "/v1/predict", {"frames": "nope"})[0] == 400


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
