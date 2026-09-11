"""Issue #7 acceptance: the ``/metrics`` endpoint and serving instrumentation.

The service is driven through its ASGI interface directly (no ``httpx`` /
``TestClient`` dependency), reusing the in-process driver pattern from
``tests/test_serve_app.py`` and extending it to carry request headers and to
return the raw body so the Prometheus ``text/plain`` response can be asserted.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
import torch

from spikeforge.observability import metrics
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.encode_spec import ENCODE_SPEC_VERSION, EncodeSpec
from spikeforge.topology import registry
from spikeforge_serve import create_app
from spikeforge_serve.app import METRICS_TOKEN_ENV

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


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset the shared registry and clear the metrics-token environment."""
    metrics.reset()
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    yield
    metrics.reset()


def _headers(
    has_body: bool, extra: Optional[Dict[str, str]]
) -> List[Tuple[bytes, bytes]]:
    """Return ASGI header pairs for a JSON request."""
    headers: List[Tuple[bytes, bytes]] = []
    if has_body:
        headers.append((b"content-type", b"application/json"))
    for name, value in (extra or {}).items():
        headers.append((name.lower().encode(), value.encode()))
    return headers


async def _http(
    app: Any,
    method: str,
    path: str,
    body: Any = None,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, bytes, Dict[str, str]]:
    """Drive ``app`` as ASGI HTTP and return ``(status, body, headers)``."""
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
    start = next(
        item
        for item in sent
        if item["type"] == "http.response.start"
    )
    chunks = [
        item["body"]
        for item in sent
        if item["type"] == "http.response.body"
    ]
    response_headers = {
        name.decode().lower(): value.decode()
        for name, value in start.get("headers", [])
    }
    return start["status"], b"".join(chunks), response_headers


def _get(
    app: Any, path: str, headers: Optional[Dict[str, str]] = None
) -> Tuple[int, bytes, Dict[str, str]]:
    """Issue a GET request against ``app``."""
    return asyncio.run(_http(app, "GET", path, headers=headers))


def _post(app: Any, path: str, body: Any) -> Tuple[int, bytes, Dict[str, str]]:
    """Issue a POST request against ``app``."""
    return asyncio.run(_http(app, "POST", path, body))


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
    }
    await app(scope, receive, send)
    return sent


def test_metrics_endpoint_exposes_prometheus_text(bundle_path: str) -> None:
    """``/metrics`` returns 200 with the expected Prometheus families."""
    app = create_app(bundle_path)
    _post(app, "/v1/predict", {"frames": [_sample(1)]})
    status, body, headers = _get(app, "/metrics")
    text = body.decode("utf-8")
    assert status == 200
    assert headers["content-type"] == (
        "text/plain; version=0.0.4; charset=utf-8"
    )
    assert "# TYPE serve_requests_total counter" in text
    assert "# TYPE serve_in_flight gauge" in text
    assert "# TYPE serve_request_seconds histogram" in text
    assert "# TYPE serve_steps_total counter" in text


def test_predict_increments_serving_metrics(bundle_path: str) -> None:
    """A predict call advances the request, step, and latency series."""
    app = create_app(bundle_path)
    _post(app, "/v1/predict", {"frames": [_sample(2)]})
    snapshot = metrics.snapshot()
    assert snapshot["counters"]["serve.requests"] == 1.0
    assert snapshot["counters"]["serve.steps"] == float(_NUM_STEPS)
    assert snapshot["timers"]["serve.request_seconds"]["count"] == 1
    assert snapshot["gauges"]["serve.in_flight"] == 0.0


def test_stream_increments_steps_and_frames(bundle_path: str) -> None:
    """A WebSocket frame advances the step and stream-frame counters."""
    app = create_app(bundle_path)
    asyncio.run(
        _websocket(app, "/v1/stream", [{"frame": _sample(3)}])
    )
    snapshot = metrics.snapshot()
    assert snapshot["counters"]["serve.stream_frames"] == 1.0
    assert snapshot["counters"]["serve.steps"] == float(_NUM_STEPS)


def test_metrics_scrape_is_not_counted(tmp_path: Path) -> None:
    """The ``/metrics`` scrape does not inflate the request counters."""
    app = create_app(str(tmp_path / "absent.spkf"))
    _get(app, "/metrics")
    assert "serve.requests" not in metrics.snapshot()["counters"]


def test_metrics_requires_a_bearer_when_configured(bundle_path: str) -> None:
    """A configured token gates ``/metrics`` on the bearer header."""
    app = create_app(bundle_path, metrics_token="secret")
    assert _get(app, "/metrics")[0] == 401
    denied = _get(app, "/metrics", headers={"authorization": "Bearer nope"})
    assert denied[0] == 401
    ok = _get(
        app, "/metrics", headers={"authorization": "Bearer secret"}
    )
    assert ok[0] == 200


def test_metrics_token_can_come_from_the_environment(
    bundle_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``SPIKEFORGE_SERVE_METRICS_TOKEN`` enables the same gate."""
    monkeypatch.setenv(METRICS_TOKEN_ENV, "envtoken")
    app = create_app(bundle_path)
    assert _get(app, "/metrics")[0] == 401
    ok = _get(
        app, "/metrics", headers={"authorization": "Bearer envtoken"}
    )
    assert ok[0] == 200
