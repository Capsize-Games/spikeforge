"""Fail-closed auth: exposed binds, login cookie, origin allowlist.

Without a configured token, ``/ws`` and ``/api/bundle/*`` stay open on
loopback binds only; an exposed bind is refused unless the operator
opts out explicitly. A configured token can be presented as the login
cookie so browsers stop sending ``?token=``, and ``/ws`` honors an
origin allowlist when one is configured.
"""

import asyncio
from typing import Any, Dict, List, Optional

import pytest
from fastapi import HTTPException, WebSocketDisconnect

from server.app import websocket_endpoint
from server.auth import (
    DASHBOARD_TOKEN_ENV,
    OPEN_ENV,
    ORIGINS_ENV,
    TOKEN_COOKIE,
)
from server.auth_routes import LoginBody, login
from server.bundle_routes import download_bundle
from spikeforge.network import model_store


class _FakeWS:
    """Enough of Starlette's ``WebSocket`` for the handshake gates."""

    def __init__(
        self, query: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        scope: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Seed handshake maps; track accept/close and outbound frames."""
        self.query_params: Dict[str, str] = query or {}
        self.headers: Dict[str, str] = headers or {}
        self.cookies: Dict[str, str] = cookies or {}
        self.scope: Dict[str, Any] = scope or {}
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


class _FakeRequest:
    """Enough of Starlette's ``Request`` for the bundle gate."""

    def __init__(
        self, headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        scope: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Seed header/cookie maps and the ASGI scope."""
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.scope: Dict[str, Any] = scope or {}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the auth env vars start unset for every test."""
    monkeypatch.delenv(DASHBOARD_TOKEN_ENV, raising=False)
    monkeypatch.delenv(OPEN_ENV, raising=False)
    monkeypatch.delenv(ORIGINS_ENV, raising=False)


def _connect(ws: _FakeWS) -> _FakeWS:
    """Run the handshake gate against ``ws`` and return it."""
    asyncio.run(websocket_endpoint(ws))  # type: ignore[arg-type]
    return ws


def _exposed() -> Dict[str, Any]:
    """Return a scope for a bind on all interfaces."""
    return {"server": ("0.0.0.0", 8000)}


def test_unset_token_accepts_loopback_bind() -> None:
    """Local development on loopback still works unauthenticated."""
    ws = _connect(_FakeWS(scope={"server": ("127.0.0.1", 8000)}))
    assert ws.accepted is True
    assert ws.close_code is None


def test_unset_token_accepts_missing_server() -> None:
    """A missing server address counts as a local bind."""
    ws = _connect(_FakeWS())
    assert ws.accepted is True
    assert ws.close_code is None


def test_unset_token_rejects_exposed_bind() -> None:
    """No token on a non-loopback bind fails the handshake."""
    ws = _connect(_FakeWS(scope=_exposed()))
    assert ws.accepted is False
    assert ws.close_code == 1008


def test_open_opt_out_allows_exposed_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit opt-out reopens an exposed bind without a token."""
    monkeypatch.setenv(OPEN_ENV, "1")
    ws = _connect(_FakeWS(scope=_exposed()))
    assert ws.accepted is True
    assert ws.close_code is None


def test_configured_token_accepts_login_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The login cookie authenticates without a query param."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    ws = _connect(_FakeWS(cookies={TOKEN_COOKIE: "secret"}))
    assert ws.accepted is True
    assert ws.close_code is None


def test_configured_token_rejects_wrong_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong cookie is rejected like a wrong query param."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    ws = _connect(_FakeWS(cookies={TOKEN_COOKIE: "wrong"}))
    assert ws.accepted is False
    assert ws.close_code == 1008


def test_disallowed_origin_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign Origin is refused when an allowlist is configured."""
    monkeypatch.setenv(ORIGINS_ENV, "https://dash.example")
    ws = _connect(_FakeWS(headers={"origin": "https://evil.example"}))
    assert ws.accepted is False
    assert ws.close_code == 1008


def test_allowed_origin_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A listed Origin passes the handshake gate."""
    monkeypatch.setenv(ORIGINS_ENV, "https://dash.example")
    ws = _connect(_FakeWS(headers={"origin": "https://dash.example"}))
    assert ws.accepted is True
    assert ws.close_code is None


def test_missing_origin_accepted_with_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native clients send no Origin and stay accepted."""
    monkeypatch.setenv(ORIGINS_ENV, "https://dash.example")
    ws = _connect(_FakeWS())
    assert ws.accepted is True
    assert ws.close_code is None


def test_login_sets_cookie_on_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The right token yields an HttpOnly auth cookie."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    response = asyncio.run(login(LoginBody(token="secret")))
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert f"{TOKEN_COOKIE}=secret" in cookie
    assert "HttpOnly" in cookie


def test_login_rejects_wrong_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong token is a 401 with no cookie."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(login(LoginBody(token="wrong")))
    assert excinfo.value.status_code == 401


def test_login_rejects_when_unset() -> None:
    """With no token configured there is nothing to log in with."""
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(login(LoginBody(token="secret")))
    assert excinfo.value.status_code == 401


def test_bundle_rejects_exposed_bind_without_token() -> None:
    """Bundle downloads fail closed on an exposed bind too."""
    request = _FakeRequest(scope=_exposed())
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(download_bundle("nope", request=request))  # type: ignore[arg-type]
    assert excinfo.value.status_code == 401
    assert "not configured" in excinfo.value.detail


def test_bundle_accepts_login_cookie(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The login cookie reaches past the bundle gate to the 404."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))
    request = _FakeRequest(cookies={TOKEN_COOKIE: "secret"})
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(download_bundle("nope", request=request))  # type: ignore[arg-type]
    assert excinfo.value.status_code == 404
