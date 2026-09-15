"""The FastAPI application factory for ``spikeforge-serve``.

``create_app`` builds the whole ASGI surface from one bundle without binding a
port, so tests and embedding callers can hand the app to any ASGI client. The
routes are intentionally thin wrappers over
:class:`~spikeforge_serve.service.ServingService`, and every
:class:`~spikeforge.serving.errors.ServingError` is mapped to an honest HTTP
status instead of a stack trace.
"""

import os
import time
from typing import Any, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.errors import ServingError
from spikeforge_serve import metrics as serve_metrics
from spikeforge_serve.errors import error_body, status_for
from spikeforge_serve.routes import install_routes
from spikeforge_serve.service import ServingService

#: Version of the HTTP surface (distinct from the package version).
SERVE_VERSION = "0.1.0"

#: Environment variable that opts ``/metrics`` into bearer auth when no token
#: is passed to :func:`create_app`. An empty value leaves the route open.
METRICS_TOKEN_ENV = "SPIKEFORGE_SERVE_METRICS_TOKEN"


class MetricsMiddleware:
    """Record request count, latency, errors, and in-flight for HTTP calls.

    A tiny pure-ASGI wrapper rather than a framework middleware, so it works
    with the dependency-free ASGI driver the tests use. The ``/metrics``
    scrape itself is passed through untouched so a scrape never inflates the
    counters it is reading.
    """

    def __init__(self, app: Any) -> None:
        """Wrap the ASGI ``app`` below this middleware."""
        self._app = app

    async def __call__(
        self, scope: Any, receive: Any, send: Any
    ) -> None:
        """Instrument one ASGI call when it is an ordinary HTTP request."""
        path = scope.get("path")
        if scope.get("type") != "http" or path == serve_metrics.METRICS_PATH:
            await self._app(scope, receive, send)
            return
        await self._call_instrumented(scope, receive, send)

    async def _call_instrumented(
        self, scope: Any, receive: Any, send: Any
    ) -> None:
        """Run one HTTP call while recording status, latency, in-flight."""
        status = {"code": 500}

        async def send_wrapper(message: Any) -> None:
            """Capture the response status before forwarding it."""
            if message.get("type") == "http.response.start":
                status["code"] = int(message.get("status", 500))
            await send(message)

        start = time.perf_counter()
        with serve_metrics.track_in_flight():
            try:
                await self._app(scope, receive, send_wrapper)
            finally:
                serve_metrics.observe_request(
                    status["code"], serve_metrics.elapsed_since(start)
                )


def _resolve_token(metrics_token: Optional[str]) -> Optional[str]:
    """Return the configured ``/metrics`` token, or None when auth is off."""
    if metrics_token is not None:
        return metrics_token or None
    return os.environ.get(METRICS_TOKEN_ENV) or None


def create_app(
    bundle: Union[str, DeploymentBundle],
    device: Union[str, Any] = "cpu",
    mode: ExecutionMode = ExecutionMode.PRODUCTION,
    service: Optional[ServingService] = None,
    title: str = "spikeforge-serve",
    version: str = SERVE_VERSION,
    metrics_token: Optional[str] = None,
) -> FastAPI:
    """Build the ASGI app that serves ``bundle`` without binding a port.

    The bundle loads lazily, so a malformed artifact becomes a typed HTTP
    error rather than a crash before the first request. When
    ``metrics_token`` (or ``SPIKEFORGE_SERVE_METRICS_TOKEN``) is set,
    ``/metrics`` requires a matching bearer header; with no token it is
    open.
    """
    return _build_app(
        bundle, device, mode, service, title, version, metrics_token
    )


def _build_app(
    bundle: Union[str, DeploymentBundle],
    device: Union[str, Any],
    mode: ExecutionMode,
    service: Optional[ServingService],
    title: str,
    version: str,
    metrics_token: Optional[str],
) -> FastAPI:
    """Construct the FastAPI app once every argument is resolved."""
    serving = service or ServingService(bundle, device=device, mode=mode)
    app = FastAPI(title=title, version=version)
    app.state.serving = serving
    app.state.metrics_token = _resolve_token(metrics_token)
    app.add_middleware(MetricsMiddleware)
    _install_handlers(app)
    install_routes(app, serving)
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
