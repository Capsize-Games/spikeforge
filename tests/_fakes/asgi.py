"""One ASGI driver for the ``spikeforge-serve`` app tests.

``test_serve_app`` and ``test_serve_metrics`` each carried their own
``_http``/``_websocket`` pair, already diverged in return shape. The
shared ``_http`` returns the raw ``(status, body, headers)`` triple and
the ``_json`` wrappers decode it, so both JSON and raw-body callers run
against the same driver.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple


def _headers(
    has_body: bool, extra: Optional[Dict[str, str]]
) -> List[Tuple[bytes, bytes]]:
    """Return ASGI header pairs for a JSON request."""
    headers: List[Tuple[bytes, bytes]] = []
    if has_body:
        headers.append((b"content-type", b"application/json"))
    for name, value in (extra or {}).items():
        headers.append(
            (name.lower().encode("utf-8"), value.encode("utf-8"))
        )
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


async def _http_json(
    app: Any,
    method: str,
    path: str,
    body: Any = None,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Any]:
    """Drive ``app`` as ASGI HTTP and return ``(status, json)``."""
    status, raw, _ = await _http(app, method, path, body, headers)
    return status, json.loads(raw or b"null")


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


def _get(
    app: Any, path: str, headers: Optional[Dict[str, str]] = None
) -> Tuple[int, bytes, Dict[str, str]]:
    """Issue a GET request against ``app``."""
    return asyncio.run(_http(app, "GET", path, headers=headers))


def _post(app: Any, path: str, body: Any) -> Tuple[int, bytes, Dict[str, str]]:
    """Issue a POST request against ``app``."""
    return asyncio.run(_http(app, "POST", path, body))


def _get_json(app: Any, path: str) -> Tuple[int, Any]:
    """Issue a GET request against ``app`` and decode its JSON body."""
    return asyncio.run(_http_json(app, "GET", path))


def _post_json(app: Any, path: str, body: Any) -> Tuple[int, Any]:
    """Issue a POST request against ``app`` and decode its JSON body."""
    return asyncio.run(_http_json(app, "POST", path, body))
