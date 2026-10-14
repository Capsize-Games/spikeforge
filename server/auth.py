"""Bearer-token gate shared by the WebSocket and bundle routes.

Mirrors ``spikeforge_serve``'s ``/metrics`` token. Production deployment
requires ``SPIKEFORGE_DASHBOARD_TOKEN``; local development may leave it
unset. Browsers can't attach a custom header to a WebSocket handshake,
so ``/ws`` also accepts the token as a query parameter or as the cookie
the login route sets -- clients that can send a header should prefer
``Authorization: Bearer`` so the token never lands in access logs.

Without a configured token the routes stay open on loopback binds only;
an exposed bind without a token is refused unless the operator opts out
explicitly with ``SPIKEFORGE_DASHBOARD_OPEN=1``.
"""

import hmac
import os
from typing import List, Optional, Tuple

#: Environment variable gating ``/ws`` and ``/api/bundle/<name>``.
DASHBOARD_TOKEN_ENV = "SPIKEFORGE_DASHBOARD_TOKEN"

#: Explicit opt-out of the fail-closed default: with no token
#: configured, an exposed bind is refused unless this is set to a
#: truthy value (``1``/``true``/``yes``/``on``).
OPEN_ENV = "SPIKEFORGE_DASHBOARD_OPEN"

#: Comma-separated ``Origin`` allowlist for ``/ws``. Unset means any
#: origin is accepted; connections without an ``Origin`` header (native
#: clients, not browsers) are always accepted.
ORIGINS_ENV = "SPIKEFORGE_DASHBOARD_ORIGINS"

#: Cookie set by ``POST /api/auth/login`` and accepted by both gates.
TOKEN_COOKIE = "spikeforge_token"

#: Hosts that count as a local bind for the fail-closed default.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def configured_token() -> Optional[str]:
    """Return the configured dashboard token, or None when auth is off."""
    return os.environ.get(DASHBOARD_TOKEN_ENV) or None


def auth_open() -> bool:
    """Return whether the operator opted out of fail-closed auth."""
    return os.environ.get(OPEN_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def bind_exposed(server: Optional[Tuple[str, int]]) -> bool:
    """Return True when the socket is bound past loopback.

    A missing server address (direct handler calls in tests) counts as
    loopback, preserving today's open local-development behavior.
    """
    if not server:
        return False
    return server[0] not in _LOOPBACK_HOSTS


def allowed_origins() -> List[str]:
    """Return the configured ``Origin`` allowlist, empty when unset."""
    raw = os.environ.get(ORIGINS_ENV, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def origin_allowed(origin: Optional[str]) -> bool:
    """Return True when a ``/ws`` handshake's origin is acceptable."""
    allowed = allowed_origins()
    if not allowed or origin is None:
        return True
    return origin in allowed


def authorized(
    query_token: Optional[str] = None,
    header: Optional[str] = None,
    cookie: Optional[str] = None,
) -> bool:
    """Return True when a caller presents the configured token.

    Always True when no token is configured (local development). Otherwise
    True when ``query_token``, the login cookie, or an
    ``Authorization: Bearer <token>`` ``header`` matches, compared with
    ``hmac.compare_digest`` to avoid a timing leak.
    """
    expected = configured_token()
    if not expected:
        return True
    if query_token and hmac.compare_digest(query_token, expected):
        return True
    if cookie and hmac.compare_digest(cookie, expected):
        return True
    if header:
        scheme, _, value = header.partition(" ")
        if scheme.lower() == "bearer" and hmac.compare_digest(
            value.strip(), expected
        ):
            return True
    return False


def request_authorized(
    server: Optional[Tuple[str, int]] = None,
    query_token: Optional[str] = None,
    header: Optional[str] = None,
    cookie: Optional[str] = None,
) -> bool:
    """Return True when a ``/ws`` or bundle request may proceed.

    With a token configured the caller must present it; without one the
    request is allowed on loopback binds (or with the explicit open
    opt-out) and refused on exposed binds.
    """
    if configured_token() is not None:
        return authorized(query_token, header, cookie)
    return not bind_exposed(server) or auth_open()
