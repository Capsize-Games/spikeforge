"""Optional bearer-token gate shared by the WebSocket and bundle routes.

Mirrors ``spikeforge_serve``'s ``/metrics`` token: unset leaves every route
open, exactly as today, so a plain ``docker compose up`` on localhost is
unaffected. Setting ``SPIKEFORGE_DASHBOARD_TOKEN`` requires a caller to
present it. Browsers can't attach a custom header to a WebSocket handshake,
and the bundle download is a plain ``<a download>`` link that can't attach
one either, so both routes also accept the token as a query parameter.
"""

import hmac
import os
from typing import Optional

#: Environment variable gating ``/ws`` and ``/api/bundle/<name>``. An empty
#: or unset value leaves both routes open.
DASHBOARD_TOKEN_ENV = "SPIKEFORGE_DASHBOARD_TOKEN"


def configured_token() -> Optional[str]:
    """Return the configured dashboard token, or None when auth is off."""
    return os.environ.get(DASHBOARD_TOKEN_ENV) or None


def authorized(
    query_token: Optional[str] = None, header: Optional[str] = None
) -> bool:
    """Return True when a caller presents the configured token.

    Always True when no token is configured. Otherwise True when either
    ``query_token`` or an ``Authorization: Bearer <token>`` ``header``
    matches, compared with ``hmac.compare_digest`` to avoid a timing leak.
    """
    expected = configured_token()
    if not expected:
        return True
    if query_token and hmac.compare_digest(query_token, expected):
        return True
    if header:
        scheme, _, value = header.partition(" ")
        if scheme.lower() == "bearer" and hmac.compare_digest(
            value.strip(), expected
        ):
            return True
    return False
