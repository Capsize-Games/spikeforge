"""Fetch the one artifact this job was dispatched to verify.

The URL is the public content-addressed download URL from the dispatch
payload (``plans/hub_accounts_plan.md`` §7.3); this module does not
authenticate to anything else and never reuses the URL beyond one GET.
"""

import urllib.error
import urllib.request
from typing import Any, Final

from hub_verify import USER_AGENT
from hub_verify.errors import FetchError

#: Matches the community-upload single-artifact cap
#: (``plans/hub_accounts_plan.md`` §5.3). A ``Content-Length`` claim is
#: never trusted alone -- the stream itself is cut off past this many bytes.
MAX_ARTIFACT_BYTES: Final[int] = 256 * 1024 * 1024

#: Refuse to hang on a stalled or hostile server.
_TIMEOUT_SECONDS = 60
_CHUNK_SIZE = 1 << 20


def _read_capped(response: Any, dest: str) -> int:
    """Stream ``response`` into ``dest``, raising past the byte cap."""
    written = 0
    with open(dest, "wb") as handle:
        while True:
            chunk = response.read(_CHUNK_SIZE)
            if not chunk:
                return written
            written += len(chunk)
            if written > MAX_ARTIFACT_BYTES:
                raise FetchError(
                    f"artifact exceeds the {MAX_ARTIFACT_BYTES} byte cap"
                )
            handle.write(chunk)


def _request(url: str) -> urllib.request.Request:
    """Return a GET for ``url`` identifying this runner."""
    return urllib.request.Request(
        url, method="GET", headers={"User-Agent": USER_AGENT}
    )


def fetch_artifact(url: str, dest: str) -> str:
    """Download ``url`` to ``dest`` and return ``dest``.

    Raises :class:`FetchError` naming the reason on any network failure or
    a payload past :data:`MAX_ARTIFACT_BYTES`, so a hostile or broken
    signed URL is reported like any other named rejection rather than an
    unhandled traceback.
    """
    try:
        with urllib.request.urlopen(
            _request(url), timeout=_TIMEOUT_SECONDS
        ) as response:
            _read_capped(response, dest)
    except FetchError:
        raise
    except urllib.error.HTTPError as error:
        raise FetchError(f"HTTP {error.code} fetching artifact") from error
    except (urllib.error.URLError, OSError) as error:
        raise FetchError(f"network fetch failed: {error}") from error
    return dest
