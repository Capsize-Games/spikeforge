"""Post the signed verification report back to the hub API.

Follows the same ``urllib``-only request pattern already used in this
monorepo (:class:`spikeforge_clients.transport.HttpTransport`) rather than
adding a new HTTP dependency: this script installs into a throwaway CI
job, but the project's stated convention is stdlib-only HTTP regardless.
"""

import json
import urllib.error
import urllib.request
from typing import Final

from hub_verify.errors import CallbackError
from hub_verify.signing import SIGNATURE_HEADER, sign_body

_TIMEOUT_SECONDS: Final[int] = 30


def _url(base_url: str, version_id: str) -> str:
    """Return the internal verification-callback URL for ``version_id``."""
    return f"{base_url.rstrip('/')}/internal/v1/verifications/{version_id}"


def post_report(
    base_url: str, version_id: str, report: dict, secret: str
) -> None:
    """POST ``report`` for ``version_id``, signed with ``secret``.

    Raises :class:`CallbackError` on any failure to deliver it -- unlike an
    artifact rejection, a delivery failure is not something this job can
    itself report to the hub, so it must fail the job loudly instead of
    silently leaving the version stuck in ``verifying``.
    """
    body = json.dumps(report, sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        _url(base_url, version_id),
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            SIGNATURE_HEADER: sign_body(body, secret),
        },
    )
    try:
        with urllib.request.urlopen(
            request, timeout=_TIMEOUT_SECONDS
        ) as reply:
            if reply.status >= 400:
                raise CallbackError(f"hub API replied {reply.status}")
    except urllib.error.HTTPError as error:
        raise CallbackError(
            f"hub API replied {error.code}: {error.read()!r}"
        ) from error
    except (urllib.error.URLError, OSError) as error:
        raise CallbackError(f"callback delivery failed: {error}") from error
