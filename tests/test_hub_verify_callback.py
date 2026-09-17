"""Posting the signed report back to the hub API."""

import json
from typing import Any
from urllib import error as urlerror

import pytest
from hub_verify import callback
from hub_verify.errors import CallbackError
from hub_verify.signing import SIGNATURE_HEADER, signature_matches


class _FakeReply:
    """A minimal stand-in for the ``urlopen`` context manager."""

    def __init__(self, status: int) -> None:
        """Record the reply's HTTP ``status``."""
        self.status = status

    def __enter__(self) -> "_FakeReply":
        """Support ``with urlopen(...) as reply``."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Nothing to release for a fake reply."""


def test_posts_a_correctly_signed_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The delivered request body is signed under the given secret."""
    sent = {}

    def _urlopen(request: Any, timeout: float) -> _FakeReply:
        sent["url"] = request.full_url
        sent["body"] = request.data
        # ``Request.add_header`` stores every key through ``.capitalize()``
        # (first character up, the rest down), so the header must be read
        # back the same way rather than by its original spelling.
        sent["signature"] = request.headers[SIGNATURE_HEADER.capitalize()]
        return _FakeReply(200)

    monkeypatch.setattr(callback.urllib.request, "urlopen", _urlopen)
    report = {"version_id": "v1", "passed": True}
    callback.post_report(
        "https://hub.spikeforge.net", "v1", report, "sekrit"
    )
    assert sent["url"] == (
        "https://hub.spikeforge.net/internal/v1/verifications/v1"
    )
    assert json.loads(sent["body"]) == report
    assert signature_matches(sent["body"], "sekrit", sent["signature"])


def test_error_status_is_a_callback_error(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4xx/5xx reply is reported by name, not swallowed."""
    monkeypatch.setattr(
        callback.urllib.request, "urlopen", lambda *a, **k: _FakeReply(500)
    )
    with pytest.raises(CallbackError, match="500"):
        callback.post_report("https://hub.spikeforge.net", "v1", {}, "s")


def test_http_error_is_a_callback_error(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """An HTTPError from urlopen is wrapped, not left to propagate raw."""

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise urlerror.HTTPError(
            "url", 404, "not found", {}, __import__("io").BytesIO(b"nope")
        )

    monkeypatch.setattr(callback.urllib.request, "urlopen", _raise)
    with pytest.raises(CallbackError, match="404"):
        callback.post_report("https://hub.spikeforge.net", "v1", {}, "s")


def test_network_error_is_a_callback_error(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection failure is a named ``CallbackError``, not a crash."""

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise urlerror.URLError("unreachable")

    monkeypatch.setattr(callback.urllib.request, "urlopen", _raise)
    with pytest.raises(CallbackError, match="callback delivery failed"):
        callback.post_report("https://hub.spikeforge.net", "v1", {}, "s")
