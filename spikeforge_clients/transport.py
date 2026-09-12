"""HTTP and WebSocket transports for the ``spikeforge-clients`` SDK.

:class:`HttpTransport` is the real-network transport: it uses the standard
library ``urllib`` for requests (so the base distribution needs no HTTP
dependency) and the optional ``websockets`` package only for streaming. Tests
and embedders use :class:`spikeforge_clients.inprocess.ASGITransport` instead.
"""

import json
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
)
from urllib import error as urlerror
from urllib import request as urlrequest

from spikeforge_clients.errors import TransportError


@dataclass(frozen=True)
class Response:
    """One HTTP response: its status and parsed JSON payload."""

    status: int
    payload: Any


class Transport(Protocol):
    """The small surface :class:`~spikeforge_clients.client.ServeClient` needs.

    Returning the status (instead of raising on non-2xx) keeps error mapping in
    one place: the client turns a status and body into a typed
    :class:`~spikeforge_clients.errors.ServiceError`.
    """

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Response:
        """Issue one request and return its status and JSON body."""
        ...

    def stream(
        self,
        path: str,
        messages: Iterable[Mapping[str, Any]],
        *,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Iterator[Mapping[str, Any]]:
        """Send ``messages`` and yield each reply envelope."""
        ...


def _encode(payload: Any) -> Optional[bytes]:
    """Return ``payload`` as JSON bytes, or None for an empty body."""
    if payload is None:
        return None
    return json.dumps(payload).encode("utf-8")


def _decode(raw: bytes) -> Any:
    """Return ``raw`` parsed as JSON, or None when it is empty/invalid."""
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def _sync_connect() -> Any:
    """Return the optional ``websockets`` sync connector, or raise."""
    try:
        from websockets.sync.client import connect
    except ImportError as error:
        raise TransportError(
            "streaming requires the 'stream' extra (websockets)"
        ) from error
    return connect


class HttpTransport:
    """Talk to a running ``spikeforge-serve`` over HTTP and WebSocket."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        token: Optional[str] = None,
    ) -> None:
        """Remember the base URL, request timeout, and bearer token."""
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._token = token

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Response:
        """Issue one HTTP request and return its status and JSON body."""
        request = urlrequest.Request(
            self._url(path),
            data=_encode(payload),
            method=method,
            headers=self._headers(payload is not None, headers),
        )
        try:
            with urlrequest.urlopen(
                request, timeout=self._timeout
            ) as reply:
                return Response(int(reply.status), _decode(reply.read()))
        except urlerror.HTTPError as error:
            return Response(int(error.code), _decode(error.read()))
        except (urlerror.URLError, OSError) as error:
            raise TransportError(
                f"{method} {path} failed: {error}"
            ) from error

    def stream(
        self,
        path: str,
        messages: Iterable[Mapping[str, Any]],
        *,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Iterator[Mapping[str, Any]]:
        """Stream ``messages`` over WebSocket and yield each reply."""
        connect = _sync_connect()
        auth = self._headers(False, headers)
        try:
            with connect(
                self._ws_url(path), additional_headers=auth
            ) as socket:
                for message in messages:
                    socket.send(json.dumps(dict(message)))
                    yield json.loads(socket.recv())
        except OSError as error:
            raise TransportError(
                f"stream {path} failed: {error}"
            ) from error

    def _url(self, path: str) -> str:
        """Return the absolute HTTP URL for ``path``."""
        return self._base_url + path

    def _ws_url(self, path: str) -> str:
        """Return the WebSocket URL for ``path`` on this service."""
        if self._base_url.startswith("https://"):
            rest = self._base_url[len("https://"):]
            return "wss://" + rest + path
        if self._base_url.startswith("http://"):
            rest = self._base_url[len("http://"):]
            return "ws://" + rest + path
        return self._base_url + path

    def _headers(
        self,
        has_body: bool,
        extra: Optional[Mapping[str, str]],
    ) -> Dict[str, str]:
        """Return the request headers, adding JSON and auth as needed."""
        headers: Dict[str, str] = {}
        if has_body:
            headers["Content-Type"] = "application/json"
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        for name, value in (extra or {}).items():
            headers[name] = value
        return headers
