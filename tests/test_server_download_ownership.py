"""Per-session download ownership and the socket hub_cancel path.

A ``cancel_download`` or ``hub_cancel`` from one session must never
disturb another session's in-flight download, and a ``hub_cancel``
arriving over a real socket must terminate an active community pull --
the socket fast path in ``server.app`` shares its logic with the
dispatched handler instead of cancelling downloads only.
"""

import asyncio
import contextlib
from typing import Any, Dict, List

from _fakes import FakeWS
from fastapi import WebSocketDisconnect

from server.app import _serve
from server.protocol_version import PROTOCOL_VERSION
from server.session import Session


class _FakeManager:
    """A download manager stand-in with an observable cancel."""

    def __init__(self, status: str = "idle") -> None:
        """Start in ``status`` with no cancel recorded."""
        self.status = status
        self.cancelled = False

    def snapshot(self) -> Dict[str, Any]:
        """Return the current fake state."""
        return {"id": "fake", "status": self.status}

    def cancel(self) -> None:
        """Record the cancel and flip an active job to cancelled."""
        self.cancelled = True
        if self.status == "downloading":
            self.status = "cancelled"




def _frame(message_type: str) -> Dict[str, Any]:
    """Return a versioned inbound frame of ``message_type``."""
    return {"protocol_version": PROTOCOL_VERSION, "type": message_type}


async def _run(
    frames: List[Dict[str, Any]], session: Session
) -> List[Dict[str, Any]]:
    """Serve ``frames`` against ``session`` until the socket drops."""
    ws = FakeWS(frames)
    with contextlib.suppress(WebSocketDisconnect):
        await _serve(ws, session)
    return ws.sent


def test_cancel_download_is_scoped_to_its_session() -> None:
    """A cancel from B leaves A's in-flight dataset download alone."""

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        own_a = _FakeManager("downloading")
        own_b = _FakeManager("idle")
        session_a = Session(loop)
        session_a.downloads = own_a
        session_b = Session(loop)
        session_b.downloads = own_b
        await _run([_frame("cancel_download")], session_b)
        assert not own_a.cancelled
        assert own_a.status == "downloading"
        assert own_b.cancelled

    asyncio.run(scenario())


def test_hub_cancel_is_scoped_to_its_session() -> None:
    """A hub_cancel from B leaves A's hub download and pull alone."""

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        download_a = _FakeManager("downloading")
        pull_a = _FakeManager("downloading")
        session_a = Session(loop)
        session_a.hub_downloads = download_a
        session_a.hub_pulls = pull_a
        session_b = Session(loop)
        await _run([_frame("hub_cancel")], session_b)
        assert not download_a.cancelled
        assert not pull_a.cancelled
        assert download_a.status == "downloading"
        assert pull_a.status == "downloading"

    asyncio.run(scenario())


def test_socket_hub_cancel_terminates_own_pull() -> None:
    """hub_cancel over the socket cancels the pull and reports it."""

    async def scenario() -> List[Dict[str, Any]]:
        session = Session(asyncio.get_running_loop())
        pull = _FakeManager("downloading")
        session.hub_pulls = pull
        sent = await _run([_frame("hub_cancel")], session)
        assert pull.cancelled
        return sent

    sent = asyncio.run(scenario())
    kinds = [item["type"] for item in sent]
    assert "hub_download_state" in kinds
    assert "hub_pull_state" in kinds
    pull_state = next(
        item for item in sent if item["type"] == "hub_pull_state"
    )
    assert pull_state["payload"]["status"] == "cancelled"


def test_socket_hub_cancel_without_pull_omits_pull_state() -> None:
    """An idle pull manager means no pull snapshot is emitted."""

    async def scenario() -> List[Dict[str, Any]]:
        session = Session(asyncio.get_running_loop())
        return await _run([_frame("hub_cancel")], session)

    sent = asyncio.run(scenario())
    kinds = [item["type"] for item in sent]
    assert kinds == ["hub_download_state"]
