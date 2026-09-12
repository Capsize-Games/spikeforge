"""A dependency-free ASGI transport that drives an app in-process.

The driver mirrors ``tests/test_serve_app.py``: it feeds ASGI http and
websocket scopes to the app returned by ``spikeforge_serve.create_app`` and
collects the responses, so the SDK can be exercised end-to-end without binding
a port or adding an HTTP dependency. WebSocket replies are buffered because
the driver runs the app to completion for the messages it was handed.
"""

import asyncio
import json
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from spikeforge_clients.transport import Response


def _pairs(
    has_body: bool, extra: Optional[Mapping[str, str]]
) -> List[Any]:
    """Return ASGI header pairs for a JSON request."""
    headers: List[Any] = []
    if has_body:
        headers.append((b"content-type", b"application/json"))
    for name, value in (extra or {}).items():
        headers.append((name.lower().encode(), value.encode()))
    return headers


def _parse(raw: bytes) -> Any:
    """Return ``raw`` parsed as JSON, or None when it is empty."""
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


class ASGITransport:
    """Drive an ASGI ``app`` with no socket, for tests and embedding."""

    def __init__(self, app: Any) -> None:
        """Wrap the ASGI ``app`` this transport sends scopes to."""
        self._app = app

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Response:
        """Run one ASGI HTTP request and return its status and JSON body."""
        return asyncio.run(self._http(method, path, payload, headers))

    async def _http(
        self,
        method: str,
        path: str,
        payload: Any,
        headers: Optional[Mapping[str, str]],
    ) -> Response:
        """Feed one http scope to the app and collect the response."""
        raw = b"" if payload is None else json.dumps(payload).encode()
        sent: List[Dict[str, Any]] = []

        async def receive() -> Dict[str, Any]:
            """Return the whole request body once."""
            return {
                "type": "http.request",
                "body": raw,
                "more_body": False,
            }

        async def send(message: Dict[str, Any]) -> None:
            """Record one response message."""
            sent.append(message)

        scope = self._http_scope(
            method, path, payload is not None, headers
        )
        await self._app(scope, receive, send)
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
        return Response(int(status), _parse(b"".join(chunks)))

    def stream(
        self,
        path: str,
        messages: Iterable[Mapping[str, Any]],
        *,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Iterator[Mapping[str, Any]]:
        """Send ``messages`` to the app and yield each decoded reply."""
        sent = asyncio.run(self._websocket(path, list(messages)))
        for message in sent:
            if message.get("type") == "websocket.send":
                yield json.loads(message["text"])

    async def _websocket(
        self, path: str, messages: List[Any]
    ) -> List[Dict[str, Any]]:
        """Feed one websocket scope to the app and collect sent frames."""
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        await queue.put({"type": "websocket.connect"})
        for message in messages:
            await queue.put(
                {
                    "type": "websocket.receive",
                    "text": json.dumps(message),
                }
            )
        await queue.put({"type": "websocket.disconnect", "code": 1000})
        sent: List[Dict[str, Any]] = []

        async def receive() -> Dict[str, Any]:
            """Return the next queued client message."""
            return await queue.get()

        async def send(message: Dict[str, Any]) -> None:
            """Record one server frame."""
            sent.append(message)

        await self._app(self._ws_scope(path), receive, send)
        return sent

    def _http_scope(
        self,
        method: str,
        path: str,
        has_body: bool,
        headers: Optional[Mapping[str, str]],
    ) -> Dict[str, Any]:
        """Return an ASGI http scope for ``method`` and ``path``."""
        return {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": _pairs(has_body, headers),
            "client": ("spikeforge-clients", 50000),
            "server": ("testserver", 80),
        }

    def _ws_scope(self, path: str) -> Dict[str, Any]:
        """Return an ASGI websocket scope for ``path``."""
        return {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "scheme": "ws",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("spikeforge-clients", 50000),
            "server": ("testserver", 80),
            "subprotocols": [],
        }
