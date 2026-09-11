"""PT-W4 contract guard: the serve schemas, the generated TS, and the app.

The client and the service cannot drift: real ``spikeforge-serve`` responses
are validated against the JSON Schemas under ``protocol/serve/`` (the same
discipline as the existing ``protocol/`` contract), and the generated
TypeScript the TS client imports must still declare every serve interface. The
TypeScript route literals are checked against the app's OpenAPI paths.
"""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from referencing import Registry, Resource
from test_serve_app import _sample, _write_bundle

from spikeforge_clients import ServeClient
from spikeforge_serve import create_app

_ROOT = Path(__file__).resolve().parent.parent
_PROTOCOL = _ROOT / "protocol"
_GENERATED = _ROOT / "client" / "src" / "protocol" / "generated.ts"
_TS_CLIENT = _ROOT / "client" / "src" / "serve" / "serveClient.ts"

_ROUTES = (
    "/health",
    "/ready",
    "/v1/bundle",
    "/v1/predict",
    "/v1/reset",
    "/v1/stream",
)


def _registry() -> Registry:
    """Register every protocol schema by ``$id`` so refs resolve offline."""
    registry = Registry()
    for path in sorted(_PROTOCOL.rglob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        registry = registry.with_resource(
            schema["$id"], Resource.from_contents(schema)
        )
    return registry


def _validate(name: str, document: Any) -> None:
    """Assert ``document`` validates against ``protocol/serve/<name>``."""
    schema = json.loads((_PROTOCOL / name).read_text(encoding="utf-8"))
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls(schema, registry=_registry()).validate(document)


@pytest.fixture()
def client(tmp_path: Path) -> ServeClient:
    """Return a client over an in-process app for a fresh bundle."""
    path = tmp_path / "model.spkf"
    _write_bundle(path)
    return ServeClient.in_process(create_app(str(path)))


def test_predict_response_matches_the_serve_schema(
    client: ServeClient,
) -> None:
    """A real ``/v1/predict`` response validates against its schema."""
    _validate(
        "serve/predict_response.schema.json",
        client.predict([_sample(1)]).to_dict(),
    )


def test_stream_messages_match_the_serve_schema(
    client: ServeClient,
) -> None:
    """Every real stream envelope validates against its schema."""
    for event in client.stream([_sample(1), _sample(2)]):
        _validate("serve/stream_message.schema.json", event.to_dict())


def test_bundle_and_reset_match_their_schemas(
    client: ServeClient,
) -> None:
    """``/v1/bundle`` and ``/v1/reset`` responses validate too."""
    _validate(
        "serve/bundle_info.schema.json", client.bundle_info().to_dict()
    )
    _validate(
        "serve/reset_response.schema.json", client.reset().to_dict()
    )


def test_generated_types_cover_the_serve_contract() -> None:
    """The regenerated TS declares every serve interface."""
    generated = _GENERATED.read_text(encoding="utf-8")
    for name in (
        "BundleInfo",
        "PredictRequest",
        "PredictResponse",
        "ResetResponse",
        "ServePrediction",
        "StreamMessage",
        "TensorValue",
    ):
        assert f"export interface {name}" in generated


def test_ts_client_uses_the_serve_routes() -> None:
    """The TS wrapper references every route the clients call."""
    text = _TS_CLIENT.read_text(encoding="utf-8")
    for route in _ROUTES:
        assert route in text


def test_client_routes_exist_in_the_app_openapi(tmp_path: Path) -> None:
    """The route literals the clients use are the app's published paths."""
    app = create_app(str(tmp_path / "model.spkf"))
    paths = set(app.openapi()["paths"])
    for route in _ROUTES:
        assert route in paths
