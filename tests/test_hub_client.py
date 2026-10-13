"""The hub API client: tokens, auth, and request shapes."""

import contextlib
import io
import json
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Tuple

import pytest

from spikeforge_hub import cache, client
from spikeforge_hub.client import HubClient
from spikeforge_hub.errors import HubApiError


@contextlib.contextmanager
def _respond(body: bytes = b"{}") -> Iterator[SimpleNamespace]:
    """Yield a minimal ``urlopen`` response stand-in."""
    yield SimpleNamespace(read=lambda: body)


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    respond: Callable[[Any], Any],
) -> List[Any]:
    """Route ``urlopen`` to ``respond``, recording every request."""
    seen: List[Any] = []

    def fake_open(request: Any, timeout: int = 0) -> Any:
        seen.append(request)
        return respond(request)

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    return seen


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the hub cache into a temporary directory, logged out."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path / "hub"))
    monkeypatch.delenv(client.TOKEN_ENV, raising=False)
    return tmp_path


def _problem(code: int, detail: str) -> urllib.error.HTTPError:
    """Return an HTTP error carrying a problem-details body."""
    body = json.dumps({
        "title": "unauthorized",
        "status": code,
        "Detail": detail,
    }).encode("utf-8")
    return urllib.error.HTTPError(
        "https://hub.test/v1/me", code, "Unauthorized", {}, io.BytesIO(body)
    )


def test_token_round_trips_with_owner_only_mode(sandbox: Path) -> None:
    """A stored token reads back, and nobody else may read the file."""
    path = client.write_token("  secret-pat\n")
    assert client.read_token() == "secret-pat"
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600


def test_token_env_wins_over_the_file(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment token is used without touching the file."""
    client.write_token("file-pat")
    monkeypatch.setenv(client.TOKEN_ENV, "env-pat")
    assert client.read_token() == "env-pat"


def test_read_token_is_none_when_logged_out(sandbox: Path) -> None:
    """No file and no variable means no token, not an error."""
    assert client.read_token() is None


def test_require_client_refuses_while_logged_out(sandbox: Path) -> None:
    """Authed calls name login instead of failing anonymously."""
    with pytest.raises(HubApiError, match="not logged in"):
        client.require_client()


def test_me_sends_bearer_auth(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``me`` GETs the profile with the token in the header."""
    body = json.dumps({"handle": "ada"}).encode("utf-8")
    seen = _serve(monkeypatch, lambda req: _respond(body))
    assert HubClient(base="https://hub.test", token="pat").me() == {
        "handle": "ada"
    }
    request = seen[0]
    assert request.full_url == "https://hub.test/v1/me"
    assert request.get_method() == "GET"
    assert request.headers["Authorization"] == "Bearer pat"


def _hub(
    monkeypatch: pytest.MonkeyPatch, replies: List[Dict[str, Any]]
) -> Tuple[HubClient, List[Any]]:
    """Return a client answering from ``replies``, recording requests."""
    seen = _serve(
        monkeypatch,
        lambda req: _respond(json.dumps(replies.pop(0)).encode("utf-8")),
    )
    return HubClient(base="https://hub.test", token="pat"), seen


def test_create_and_commit_hit_their_routes(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reserve and commit POST their documented routes."""
    hub, seen = _hub(monkeypatch, [
        {"upload_id": "u1"}, {"state": "published"},
    ])
    assert hub.create_upload({"name": "m"}) == {"upload_id": "u1"}
    assert hub.commit_upload("u1") == {"state": "published"}
    assert [request.get_method() for request in seen] == ["POST", "POST"]
    assert seen[0].full_url.endswith("/v1/uploads")
    assert seen[1].full_url.endswith("/v1/uploads/u1/commit")


def test_put_sends_the_reserved_bytes(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The content PUT carries the bytes to the reservation's route."""
    hub, seen = _hub(monkeypatch, [{"state": "received"}])
    assert hub.put_content("u1", b"bytes") == {"state": "received"}
    assert seen[0].get_method() == "PUT"
    assert seen[0].full_url.endswith("/v1/uploads/u1/content")
    assert seen[0].data == b"bytes"


def test_refusal_surfaces_the_server_detail(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A problem-details refusal reports its human reason and status."""

    def refuse(request: Any) -> Any:
        raise _problem(401, "this token was revoked")

    _serve(monkeypatch, refuse)
    with pytest.raises(HubApiError) as caught:
        HubClient(base="https://hub.test", token="pat").me()
    assert caught.value.status == 401
    assert "this token was revoked" in str(caught.value)


def test_unreachable_server_names_the_url(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead network is a named failure, not a bare URLError."""

    def offline(request: Any) -> Any:
        raise urllib.error.URLError("no route to host")

    _serve(monkeypatch, offline)
    with pytest.raises(HubApiError, match="cannot reach"):
        HubClient(base="https://hub.test", token="pat").me()


def test_unreadable_reply_is_not_a_network_error(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200 with garbage names the server, not the connection."""
    _serve(monkeypatch, lambda req: _respond(b"{nope"))
    with pytest.raises(HubApiError, match="answered unreadably"):
        HubClient(base="https://hub.test", token="pat").me()
