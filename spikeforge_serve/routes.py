"""Route registration for the ``spikeforge-serve`` FastAPI app.

Split out of ``app`` so the app-factory module stays within the file-length
limit. Every handler here is a thin wrapper over
:class:`~spikeforge_serve.service.ServingService`; framework and transport
concerns (auth, JSON parsing, WebSocket framing) stay in this module so
``ServingService`` itself never depends on FastAPI.
"""

import hmac
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from spikeforge.observability import prometheus
from spikeforge.serving.errors import ServingError
from spikeforge_serve import metrics as serve_metrics
from spikeforge_serve.errors import error_body
from spikeforge_serve.route_helpers import (
    _body,
    _bundle_info,
    _optional_body,
    _predict_response,
    _reset_response,
    _stream_reply,
)
from spikeforge_serve.service import DEFAULT_SESSION, ServingService


def _authorized(header: Optional[str], token: Optional[str]) -> bool:
    """Return True when ``header`` carries the expected bearer ``token``."""
    if not token:
        return True
    if not header:
        return False
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(value.strip(), token)


def install_routes(app: FastAPI, serving: ServingService) -> None:
    """Register the health, bundle, predict, reset, and stream routes."""
    _add_health_route(app)
    _add_metrics_route(app)
    _add_ready_route(app, serving)
    _add_bundle_route(app, serving)
    _add_predict_route(app, serving)
    _add_reset_route(app, serving)
    _add_stream_info_route(app)
    _add_stream_route(app, serving)


def _add_health_route(app: FastAPI) -> None:
    """Register the liveness probe."""

    @app.get("/health")
    async def health() -> Dict[str, str]:
        """Liveness probe: the process is up."""
        return {"status": "ok"}


def _add_metrics_route(app: FastAPI) -> None:
    """Register the Prometheus scrape, gated by bearer auth."""

    @app.get("/metrics")
    async def metrics_endpoint(request: Request) -> Response:
        """Expose the shared registry as Prometheus text, gated by auth."""
        token = getattr(request.app.state, "metrics_token", None)
        if not _authorized(request.headers.get("authorization"), token):
            return Response(
                content="unauthorized\n",
                status_code=401,
                media_type="text/plain",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return Response(
            content=serve_metrics.render_text(),
            media_type=prometheus.CONTENT_TYPE,
        )


def _add_ready_route(app: FastAPI, serving: ServingService) -> None:
    """Register the readiness probe."""

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


def _add_bundle_route(app: FastAPI, serving: ServingService) -> None:
    """Register the bundle metadata route."""

    @app.get("/v1/bundle")
    async def bundle_info() -> Dict[str, Any]:
        """Return the loaded bundle's self-describing metadata."""
        return _bundle_info(serving.bundle)


def _add_predict_route(app: FastAPI, serving: ServingService) -> None:
    """Register the batch prediction route."""

    @app.post("/v1/predict")
    async def predict(request: Request) -> Dict[str, Any]:
        """Stream the request's frames through a session."""
        payload = await _body(request)
        return _predict_response(serving, payload)


def _add_reset_route(app: FastAPI, serving: ServingService) -> None:
    """Register the session-reset route."""

    @app.post("/v1/reset")
    async def reset(request: Request) -> Dict[str, Any]:
        """Clear a session's temporal state."""
        payload = await _optional_body(request)
        return _reset_response(serving, payload)


def _add_stream_info_route(app: FastAPI) -> None:
    """Register the WebSocket protocol description route."""

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


def _add_stream_route(app: FastAPI, serving: ServingService) -> None:
    """Register the WebSocket streaming route."""

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket) -> None:
        """Stream one step per frame, with on-demand reset."""
        await ws.accept()
        session_id = DEFAULT_SESSION
        try:
            while True:
                message = await ws.receive_json()
                serve_metrics.count_stream_frames()
                session_id, reply = _stream_reply(
                    serving, session_id, message
                )
                await ws.send_json(reply)
        except WebSocketDisconnect:
            return
