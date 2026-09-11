"""The FastAPI application factory for ``spikeforge-serve``.

``create_app`` builds the whole ASGI surface from one bundle without binding a
port, so tests and embedding callers can hand the app to any ASGI client. The
routes are intentionally thin wrappers over
:class:`~spikeforge_serve.service.ServingService`, and every
:class:`~spikeforge.serving.errors.ServingError` is mapped to an honest HTTP
status instead of a stack trace.
"""

from typing import Any, Dict, Mapping, Optional, Tuple, Union

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse

from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.errors import ServingError
from spikeforge_serve.errors import error_body, status_for
from spikeforge_serve.payloads import (
    encoded_flag,
    frames_from,
    prediction_json,
    session_id_from,
)
from spikeforge_serve.service import DEFAULT_SESSION, ServingService

#: Version of the HTTP surface (distinct from the package version).
SERVE_VERSION = "0.1.0"


def create_app(
    bundle: Union[str, DeploymentBundle],
    device: Union[str, Any] = "cpu",
    mode: ExecutionMode = ExecutionMode.PRODUCTION,
    service: Optional[ServingService] = None,
    title: str = "spikeforge-serve",
    version: str = SERVE_VERSION,
) -> FastAPI:
    """Build the ASGI app that serves ``bundle`` without binding a port.

    Tests hand the returned app to any ASGI client. The bundle is loaded
    lazily, so a missing or malformed artifact is reported as a typed HTTP
    error rather than crashing the process before the first request.
    """
    serving = service or ServingService(bundle, device=device, mode=mode)
    app = FastAPI(title=title, version=version)
    app.state.serving = serving
    _install_handlers(app)
    _install_routes(app, serving)
    return app


def _install_handlers(app: FastAPI) -> None:
    """Register the mapping from serving errors to HTTP responses."""

    @app.exception_handler(ServingError)
    async def _serving_error(
        _request: Request, error: ServingError
    ) -> JSONResponse:
        """Return the taxonomy-mapped status and a typed error body."""
        return JSONResponse(
            status_code=status_for(error), content=error_body(error)
        )


def _install_routes(app: FastAPI, serving: ServingService) -> None:
    """Register the health, bundle, predict, reset, and stream routes."""

    @app.get("/health")
    async def health() -> Dict[str, str]:
        """Liveness probe: the process is up."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> Any:
        """Readiness probe: the bundle loaded and sessions can run."""
        try:
            _ = serving.bundle
        except ServingError as error:
            return JSONResponse(
                status_code=503, content=error_body(error)
            )
        return {"status": "ready"}

    @app.get("/v1/bundle")
    async def bundle_info() -> Dict[str, Any]:
        """Return the loaded bundle's self-describing metadata."""
        return _bundle_info(serving.bundle)

    @app.post("/v1/predict")
    async def predict(request: Request) -> Dict[str, Any]:
        """Stream the request's frames through a session."""
        payload = await _body(request)
        try:
            frames = frames_from(payload)
            encoded = encoded_flag(payload)
            session_id = session_id_from(payload)
            predictions = serving.predict(
                frames, session_id=session_id, encoded=encoded
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=400, detail=str(error)
            ) from None
        return {
            "session_id": session_id,
            "steps": serving.session(session_id).steps,
            "predictions": [
                prediction_json(item) for item in predictions
            ],
        }

    @app.post("/v1/reset")
    async def reset(request: Request) -> Dict[str, Any]:
        """Clear a session's temporal state."""
        payload = await _optional_body(request)
        try:
            session_id = session_id_from(payload)
            steps = serving.reset(session_id)
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=400, detail=str(error)
            ) from None
        return {"session_id": session_id, "steps": steps}

    @app.get("/v1/stream")
    async def stream_info() -> Dict[str, Any]:
        """Describe the WebSocket streaming protocol."""
        return {
            "protocol": "websocket",
            "path": "/v1/stream",
            "frames": "raw samples or pre-encoded frames",
            "messages": {
                "predict": {"frame": "<sample or frame>"},
                "reset": {"reset": True},
            },
        }

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket) -> None:
        """Stream one step per frame, with on-demand reset."""
        await ws.accept()
        session_id = DEFAULT_SESSION
        try:
            while True:
                message = await ws.receive_json()
                session_id, reply = _stream_reply(
                    serving, session_id, message
                )
                await ws.send_json(reply)
        except WebSocketDisconnect:
            return


async def _body(request: Request) -> Any:
    """Return the request's JSON body, or raise a 400."""
    try:
        return await request.json()
    except Exception as error:  # any parse failure is a client error
        raise HTTPException(
            status_code=400,
            detail=f"request body is not valid JSON: {error}",
        ) from None


async def _optional_body(request: Request) -> Dict[str, Any]:
    """Return the JSON object body, treating an empty body as ``{}``."""
    raw = await request.body()
    if not raw:
        return {}
    return await _body(request)


def _stream_reply(
    serving: ServingService, session_id: str, message: Any
) -> Tuple[str, Dict[str, Any]]:
    """Return ``(session_id, reply)`` for one stream message."""
    if not isinstance(message, Mapping):
        return session_id, _stream_error(
            "stream message must be a JSON object"
        )
    try:
        if "session_id" in message:
            session_id = session_id_from(message)
        if message.get("reset"):
            steps = serving.reset(session_id)
            return session_id, {
                "type": "reset",
                "payload": {"session_id": session_id, "steps": steps},
            }
        if "frame" not in message:
            return session_id, {
                "type": "session",
                "payload": {"session_id": session_id},
            }
        encoded = encoded_flag(message)
        prediction = serving.predict(
            [message["frame"]], session_id=session_id, encoded=encoded
        )[0]
        payload = prediction_json(prediction)
        payload["session_id"] = session_id
        payload["steps"] = serving.session(session_id).steps
        return session_id, {"type": "prediction", "payload": payload}
    except (TypeError, ValueError) as error:
        return session_id, _stream_error(str(error))


def _stream_error(detail: str) -> Dict[str, Any]:
    """Return a stream error envelope for ``detail``."""
    return {
        "type": "error",
        "payload": {"type": "bad_request", "message": detail},
    }


def _bundle_info(bundle: DeploymentBundle) -> Dict[str, Any]:
    """Return the self-describing metadata a client needs for the model."""
    manifest = bundle.manifest
    spec = bundle.encode_spec()
    encode = spec.to_dict()
    return {
        "format": manifest.get("format"),
        "version": manifest.get("version"),
        "path": bundle.path,
        "topology": manifest.get("topology"),
        "topology_params": dict(manifest.get("topology_params") or {}),
        "spec": manifest.get("spec"),
        "protocol_version": manifest.get("protocol_version"),
        "encode_spec_version": manifest.get("encode_spec_version"),
        "encode_spec": encode,
        "encode_digest": spec.digest(),
        "input_size": encode["input_size"],
        "num_classes": manifest.get("num_classes"),
        "num_steps": manifest.get("num_steps"),
        "label_map": dict(manifest.get("label_map") or {}),
        "expected_metrics": dict(manifest.get("expected_metrics") or {}),
        "library_versions": dict(manifest.get("library_versions") or {}),
    }
