"""``/ws``'s auth gate: rejected before ``accept()`` when it's missing.

A token configured via ``SPIKEFORGE_DASHBOARD_TOKEN`` and missing or wrong
on the connection is rejected outright, exactly as an unset token leaves
today's handshake unaffected. Drives ``websocket_endpoint`` directly against
a fake WebSocket, mirroring ``tests/test_protocol_negotiation.py``'s
``FakeWS`` rather than standing up a real ASGI/``TestClient`` stack.
"""

import asyncio

import pytest
from _fakes import FakeWS

from server.app import websocket_endpoint
from server.auth import DASHBOARD_TOKEN_ENV


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the token env var starts unset for every test."""
    monkeypatch.delenv(DASHBOARD_TOKEN_ENV, raising=False)


def test_unset_token_accepts_any_connection() -> None:
    """No env var means the handshake behaves exactly as before."""
    ws = FakeWS()
    asyncio.run(websocket_endpoint(ws))
    assert ws.accepted is True
    assert ws.close_code is None


def test_configured_token_rejects_a_missing_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured token with none supplied is rejected before accept()."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    ws = FakeWS()
    asyncio.run(websocket_endpoint(ws))
    assert ws.accepted is False
    assert ws.close_code == 1008


def test_configured_token_rejects_the_wrong_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong query-param token is rejected the same way."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    ws = FakeWS(query={"token": "wrong"})
    asyncio.run(websocket_endpoint(ws))
    assert ws.accepted is False
    assert ws.close_code == 1008


def test_configured_token_accepts_the_matching_query_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The right query-param token lets the handshake through."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    ws = FakeWS(query={"token": "secret"})
    asyncio.run(websocket_endpoint(ws))
    assert ws.accepted is True
    assert ws.close_code is None


def test_configured_token_accepts_the_matching_bearer_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching Authorization: Bearer header also lets it through."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    ws = FakeWS(headers={"authorization": "Bearer secret"})
    asyncio.run(websocket_endpoint(ws))
    assert ws.accepted is True
    assert ws.close_code is None
