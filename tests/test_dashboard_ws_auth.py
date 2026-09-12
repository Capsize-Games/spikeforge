"""``/ws``'s auth gate: rejected before ``accept()`` when it's missing.

A token configured via ``SPIKEFORGE_DASHBOARD_TOKEN`` and missing or wrong
on the connection is rejected outright, exactly as an unset token leaves
today's handshake unaffected. Drives ``websocket_endpoint`` directly against
a fake WebSocket, mirroring ``tests/test_protocol_negotiation.py``'s
``_FakeWS`` rather than standing up a real ASGI/``TestClient`` stack.
"""

import asyncio
from typing import Any, Dict, List, Optional

import pytest
from fastapi import WebSocketDisconnect

from server.app import websocket_endpoint
from server.auth import DASHBOARD_TOKEN_ENV


class _FakeWS:
    """Enough of Starlette's ``WebSocket`` for the auth gate and one loop."""

    def __init__(
        self, query: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Seed query/header maps; track accept/close and outbound frames."""
        self.query_params: Dict[str, str] = query or {}
        self.headers: Dict[str, str] = headers or {}
        self.accepted = False
        self.close_code: Optional[int] = None
        self.sent: List[Dict[str, Any]] = []

    async def accept(self) -> None:
        """Record that the handshake completed."""
        self.accepted = True

    async def close(self, code: int = 1000) -> None:
        """Record the close code the handler chose."""
        self.close_code = code

    async def receive_json(self) -> Dict[str, Any]:
        """Disconnect immediately -- these tests only cover the gate."""
        raise WebSocketDisconnect()

    async def send_json(self, message: Dict[str, Any]) -> None:
        """Record one outbound message."""
        self.sent.append(message)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the token env var starts unset for every test."""
    monkeypatch.delenv(DASHBOARD_TOKEN_ENV, raising=False)


def test_unset_token_accepts_any_connection() -> None:
    """No env var means the handshake behaves exactly as before."""
    ws = _FakeWS()
    asyncio.run(websocket_endpoint(ws))  # type: ignore[arg-type]
    assert ws.accepted is True
    assert ws.close_code is None


def test_configured_token_rejects_a_missing_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured token with none supplied is rejected before accept()."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    ws = _FakeWS()
    asyncio.run(websocket_endpoint(ws))  # type: ignore[arg-type]
    assert ws.accepted is False
    assert ws.close_code == 1008


def test_configured_token_rejects_the_wrong_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong query-param token is rejected the same way."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    ws = _FakeWS(query={"token": "wrong"})
    asyncio.run(websocket_endpoint(ws))  # type: ignore[arg-type]
    assert ws.accepted is False
    assert ws.close_code == 1008


def test_configured_token_accepts_the_matching_query_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The right query-param token lets the handshake through."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    ws = _FakeWS(query={"token": "secret"})
    asyncio.run(websocket_endpoint(ws))  # type: ignore[arg-type]
    assert ws.accepted is True
    assert ws.close_code is None


def test_configured_token_accepts_the_matching_bearer_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching Authorization: Bearer header also lets it through."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    ws = _FakeWS(headers={"authorization": "Bearer secret"})
    asyncio.run(websocket_endpoint(ws))  # type: ignore[arg-type]
    assert ws.accepted is True
    assert ws.close_code is None
