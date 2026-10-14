"""Token login: trade the dashboard token for an HttpOnly cookie.

The dashboard's WebSocket handshake and bundle download link cannot
attach an ``Authorization`` header, and passing the token as ``?token=``
leaks it into proxy logs, browser history, and referers. A client POSTs
the token here once (in the request body, which access logs don't
record) and the browser attaches the resulting cookie to ``/ws`` and
``/api/bundle/*`` from then on.
"""

import hmac

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from server.auth import TOKEN_COOKIE, configured_token

router = APIRouter()


class LoginBody(BaseModel):
    """The dashboard token submitted to the login route."""

    token: str


@router.post("/api/auth/login")
async def login(body: LoginBody) -> JSONResponse:
    """Set the auth cookie when ``body`` carries the configured token."""
    expected = configured_token()
    if not expected or not hmac.compare_digest(body.token, expected):
        raise HTTPException(401, detail="missing or invalid token")
    response = JSONResponse({"status": "ok"})
    response.set_cookie(
        TOKEN_COOKIE, body.token, httponly=True, samesite="lax", path="/"
    )
    return response
