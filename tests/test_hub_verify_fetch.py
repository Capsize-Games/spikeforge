"""Fetching the dispatched artifact, capped and named on failure."""

import io
from typing import Any
from urllib import error as urlerror

import pytest
from hub_verify import fetch
from hub_verify.errors import FetchError


class _FakeResponse:
    """A minimal stand-in for ``http.client.HTTPResponse``."""

    def __init__(self, payload: bytes) -> None:
        """Wrap ``payload`` behind a chunked ``read``."""
        self._buffer = io.BytesIO(payload)

    def read(self, size: int) -> bytes:
        """Return up to ``size`` bytes, matching the real response API."""
        return self._buffer.read(size)

    def __enter__(self) -> "_FakeResponse":
        """Support the ``with urlopen(...) as response`` pattern."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Nothing to release for an in-memory buffer."""


def test_fetch_writes_the_payload(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normal response is streamed to ``dest`` unchanged."""
    payload = b"spkf-bytes"
    monkeypatch.setattr(
        fetch.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(payload),
    )
    dest = str(tmp_path / "artifact.spkf")
    fetch.fetch_artifact("https://cdn.example/x", dest)
    with open(dest, "rb") as handle:
        assert handle.read() == payload


def test_fetch_refuses_past_the_byte_cap(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A payload larger than the cap is refused, not buffered in full."""
    monkeypatch.setattr(fetch, "MAX_ARTIFACT_BYTES", 4)
    monkeypatch.setattr(
        fetch.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(b"way too much data"),
    )
    with pytest.raises(FetchError, match="byte cap"):
        fetch.fetch_artifact("https://cdn.example/x", str(tmp_path / "a"))


def test_http_error_is_a_named_fetch_error(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-2xx signed-URL response is a named, not a generic, failure."""

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise urlerror.HTTPError("url", 403, "forbidden", {}, None)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", _raise)
    with pytest.raises(FetchError, match="HTTP 403"):
        fetch.fetch_artifact("https://cdn.example/x", str(tmp_path / "a"))


def test_network_error_is_a_named_fetch_error(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection failure is reported by name, not as a raw traceback."""

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise urlerror.URLError("no route to host")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", _raise)
    with pytest.raises(FetchError, match="network fetch failed"):
        fetch.fetch_artifact("https://cdn.example/x", str(tmp_path / "a"))
