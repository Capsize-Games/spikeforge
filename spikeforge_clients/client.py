"""The typed ``ServeClient`` for the ``spikeforge-serve`` HTTP API."""

from typing import Any, Dict, Iterable, Iterator, Optional

from spikeforge_clients.bundle import BundleInfo
from spikeforge_clients.config import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    ClientConfig,
)
from spikeforge_clients.errors import ServiceError, StreamError
from spikeforge_clients.health import Health, Readiness
from spikeforge_clients.inprocess import ASGITransport
from spikeforge_clients.prediction import (
    PredictResponse,
    ResetResult,
)
from spikeforge_clients.stream import StreamEvent
from spikeforge_clients.transport import (
    HttpTransport,
    Response,
    Transport,
)


class ServeClient:
    """A typed client for the ``spikeforge-serve`` REST and stream API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        transport: Optional[Transport] = None,
        config: Optional[ClientConfig] = None,
        timeout: Optional[float] = None,
        token: Optional[str] = None,
    ) -> None:
        """Build a client over ``base_url`` or an injected ``transport``."""
        settings = config or ClientConfig(
            base_url=base_url or DEFAULT_BASE_URL,
            timeout=DEFAULT_TIMEOUT if timeout is None else timeout,
            token=token,
        )
        self._config = settings
        self._transport = transport or HttpTransport(
            settings.base_url, settings.timeout, settings.token
        )

    @classmethod
    def in_process(cls, app: Any, **kwargs: Any) -> "ServeClient":
        """Build a client that drives an ASGI ``app`` with no network."""
        return cls(transport=ASGITransport(app), **kwargs)

    @property
    def config(self) -> ClientConfig:
        """Return the connection settings this client was built with."""
        return self._config

    def health(self) -> Health:
        """Return the service's liveness payload."""
        response = self._transport.request("GET", "/health")
        return Health.from_json(self._ok(response))

    def ready(self) -> Readiness:
        """Return readiness without raising when the bundle is not loaded."""
        response = self._transport.request("GET", "/ready")
        return Readiness.from_response(response.status, response.payload)

    def bundle_info(self) -> BundleInfo:
        """Return the loaded bundle's self-describing metadata."""
        response = self._transport.request("GET", "/v1/bundle")
        return BundleInfo.from_json(self._ok(response))

    def predict(
        self,
        frames: Iterable[Any],
        *,
        encoded: bool = False,
        session_id: Optional[str] = None,
    ) -> PredictResponse:
        """Advance a session over ``frames``; one prediction per frame."""
        body = self._body(frames, encoded, session_id)
        response = self._transport.request(
            "POST", "/v1/predict", payload=body
        )
        return PredictResponse.from_json(self._ok(response))

    def reset(self, session_id: Optional[str] = None) -> ResetResult:
        """Clear a session's temporal state and return its step count."""
        body: Dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        response = self._transport.request(
            "POST", "/v1/reset", payload=body
        )
        return ResetResult.from_json(self._ok(response))

    def stream(
        self,
        frames: Iterable[Any],
        *,
        encoded: bool = False,
        session_id: Optional[str] = None,
    ) -> Iterator[StreamEvent]:
        """Send each frame in ``frames`` and yield one reply per frame."""
        messages = (
            self._message(frame, encoded, session_id) for frame in frames
        )
        for reply in self._transport.stream("/v1/stream", messages):
            event = StreamEvent.from_json(reply)
            if event.type == "error":
                raise StreamError(
                    str(event.payload.get("type", "error")),
                    event.error or "stream error",
                )
            yield event

    def _body(
        self,
        frames: Iterable[Any],
        encoded: bool,
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        """Return the REST predict body for ``frames``."""
        body: Dict[str, Any] = {"frames": list(frames)}
        if encoded:
            body["encoded"] = True
        if session_id:
            body["session_id"] = session_id
        return body

    def _message(
        self,
        frame: Any,
        encoded: bool,
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        """Return one stream message for ``frame``."""
        message: Dict[str, Any] = {"frame": frame}
        if encoded:
            message["encoded"] = True
        if session_id:
            message["session_id"] = session_id
        return message

    def _ok(self, response: Response) -> Any:
        """Return the payload, raising ``ServiceError`` on a non-2xx."""
        if response.status >= 400:
            raise ServiceError.from_response(
                response.status, response.payload
            )
        return response.payload
