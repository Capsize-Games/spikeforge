"""Thin HTTP client for the community hub API.

Browsing and downloading need no account; publishing does, with a personal
access token minted in the browser and pasted into ``spikeforge-hub
login``. The token lives in ``$SPIKEFORGE_HUB_TOKEN`` when set, else in a
mode-``0600`` file beside the hub cache -- the same cache-directory
precedent ``huggingface_hub`` uses for its own token.

Only ``urllib`` is used, so this module adds no dependency to the
``spikeforge-hub`` distribution. Server refusals arrive as RFC 9457
problem details; their human ``detail`` is surfaced, never the bare
status code.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from spikeforge.config import HUB_API_BASE
from spikeforge_hub import cache
from spikeforge_hub.errors import HubApiError

#: Environment variable carrying a hub personal access token.
TOKEN_ENV = "SPIKEFORGE_HUB_TOKEN"
#: Token filename inside the hub cache directory.
TOKEN_FILE_NAME = "token"
#: Seconds to wait for one API call before giving up.
REQUEST_TIMEOUT = 30


def token_path() -> str:
    """Return the path of the stored hub token file."""
    return os.path.join(cache.cache_root(), TOKEN_FILE_NAME)


def read_token() -> Optional[str]:
    """Return the configured hub token, or ``None`` when logged out.

    The environment wins over the file, so one-shot and CI callers never
    have to write credentials to disk.
    """
    env = os.environ.get(TOKEN_ENV, "").strip()
    if env:
        return env
    try:
        with open(token_path(), encoding="utf-8") as handle:
            value = handle.read().strip()
    except OSError:
        return None
    return value or None


def write_token(token: str) -> str:
    """Store ``token`` with owner-only permissions and return its path."""
    path = token_path()
    cache.ensure_dir(cache.cache_root())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise
    with handle:
        handle.write(token.strip() + "\n")
    os.chmod(path, 0o600)
    return path


class HubClient:
    """Authenticated calls against the community hub API."""

    def __init__(
        self, base: Optional[str] = None, token: Optional[str] = None
    ) -> None:
        """Point at ``base`` with ``token`` (both default when omitted)."""
        self._base = (base or HUB_API_BASE).rstrip("/")
        self._token = token

    @property
    def base(self) -> str:
        """Return the API base URL this client talks to."""
        return self._base

    def _headers(self) -> Dict[str, str]:
        """Return the authorized headers, refusing anonymous calls."""
        if not self._token:
            raise HubApiError(
                "auth",
                "not logged in; run `spikeforge-hub login` or set "
                f"{TOKEN_ENV}",
            )
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token}",
        }

    def _send(
        self, request: urllib.request.Request, path: str
    ) -> Any:
        """Send one request and return its decoded JSON body."""
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise _refused(path, exc) from exc
        except OSError as exc:
            raise HubApiError(
                path, f"cannot reach {request.full_url}: {exc}"
            ) from exc
        return _decode(path, raw)

    def _request(
        self, method: str, path: str, body: Optional[bytes] = None
    ) -> Any:
        """Send one JSON call and return its decoded body."""
        headers = self._headers()
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self._base + path, data=body, method=method, headers=headers
        )
        return self._send(request, path)

    def me(self) -> Dict[str, Any]:
        """Return the signed-in account's profile and storage position."""
        return self._request("GET", "/v1/me")

    def create_upload(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Reserve an upload slot, or refuse with the limiting rule."""
        payload = json.dumps(intent).encode("utf-8")
        return self._request("POST", "/v1/uploads", body=payload)

    def put_content(self, upload_id: str, data: bytes) -> Dict[str, Any]:
        """Send the reserved bytes for ``upload_id``."""
        headers = self._headers()
        headers["Content-Type"] = "application/octet-stream"
        request = urllib.request.Request(
            self._base + f"/v1/uploads/{upload_id}/content",
            data=data,
            method="PUT",
            headers=headers,
        )
        return self._send(request, f"/v1/uploads/{upload_id}/content")

    def commit_upload(self, upload_id: str) -> Dict[str, Any]:
        """Publish a received upload, or send it for verification."""
        return self._request("POST", f"/v1/uploads/{upload_id}/commit")


def require_client(token: Optional[str] = None) -> HubClient:
    """Return an authenticated client, or refuse while logged out."""
    found = token if token is not None else read_token()
    if not found:
        raise HubApiError(
            "auth",
            "not logged in; run `spikeforge-hub login` or set "
            f"{TOKEN_ENV}",
        )
    return HubClient(token=found)


def _decode(path: str, raw: str) -> Any:
    """Decode one reply body, naming the server when it is garbage."""
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise HubApiError(
            path, f"the server answered unreadably: {exc}"
        ) from exc


def _refused(path: str, exc: urllib.error.HTTPError) -> HubApiError:
    """Return the typed refusal for an answered-but-failed API call."""
    return HubApiError(path, f"HTTP {exc.code}: {_detail(exc)}", exc.code)


def _detail(exc: urllib.error.HTTPError) -> str:
    """Return the server's human reason, or the raw body when unreadable."""
    try:
        raw = exc.read().decode("utf-8", "replace")
    except OSError:
        return exc.reason or "request failed"
    try:
        problem = json.loads(raw)
    except ValueError:
        return raw.strip() or (exc.reason or "request failed")
    if isinstance(problem, dict):
        for key in ("Detail", "detail", "title"):
            value = problem.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return raw.strip() or (exc.reason or "request failed")
