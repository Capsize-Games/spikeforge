"""Strict ``protocol_version`` negotiation on the inbound WebSocket path.

Phase 2 retires the Phase 1 legacy window: a message that omits
``protocol_version`` is now rejected exactly like one whose MAJOR differs.
"""

import asyncio
import contextlib
from typing import Any, Dict, List

from fastapi import WebSocketDisconnect

from server.app import VERSION_MISMATCH, _serve, _version_error
from server.session import Session

_GOOD: Dict[str, Any] = {"protocol_version": "1.0", "type": "stats"}
_MINOR: Dict[str, Any] = {"protocol_version": "1.4", "type": "stats"}
_MISSING: Dict[str, Any] = {"type": "stats"}
_WRONG_MAJOR: Dict[str, Any] = {"protocol_version": "2.0", "type": "stats"}


def test_a_matching_major_is_accepted() -> None:
    """A same-MAJOR version passes, whether or not it is an exact match."""
    assert _version_error(_GOOD) is None
    assert _version_error(_MINOR) is None


def test_a_missing_version_is_a_mismatch() -> None:
    """A message without protocol_version is rejected (Phase 2 onward)."""
    error = _version_error(_MISSING)
    assert error is not None
    assert error["code"] == VERSION_MISMATCH
    assert error["client"] is None
    assert error["server"] == "1.0"


def test_a_mismatched_major_is_a_mismatch() -> None:
    """A different MAJOR component is rejected with the same code."""
    error = _version_error(_WRONG_MAJOR)
    assert error is not None
    assert error["code"] == VERSION_MISMATCH
    assert error["client"] == "2.0"


class _FakeWS:
    """Queue inbound frames, then disconnect; capture outbound frames."""

    def __init__(self, incoming: List[Dict[str, Any]]) -> None:
        """Seed the inbound queue and an empty send log."""
        self._incoming = list(incoming)
        self.sent: List[Dict[str, Any]] = []

    async def receive_json(self) -> Dict[str, Any]:
        """Return the next queued frame, or disconnect when drained."""
        if not self._incoming:
            raise WebSocketDisconnect()
        return self._incoming.pop(0)

    async def send_json(self, message: Dict[str, Any]) -> None:
        """Record one outbound message."""
        self.sent.append(message)


def _pump(incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run ``_serve`` over the queued frames and return what it emitted."""
    async def _run() -> List[Dict[str, Any]]:
        ws = _FakeWS(incoming)
        session = Session(asyncio.get_running_loop())
        with contextlib.suppress(WebSocketDisconnect):
            await _serve(ws, session)
        return ws.sent

    return asyncio.run(_run())


def test_serve_rejects_a_missing_version_with_an_error_frame() -> None:
    """The wire frame is a ``type: error`` carrying the mismatch payload."""
    sent = _pump([dict(_MISSING)])
    assert [message["type"] for message in sent] == ["error"]
    assert sent[0]["payload"]["code"] == VERSION_MISMATCH


def test_serve_accepts_a_current_version() -> None:
    """A version-1.0 frame is queued, not rejected."""
    sent = _pump([dict(_GOOD)])
    assert sent == []
