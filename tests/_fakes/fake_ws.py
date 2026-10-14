"""A fake WebSocket covering the three shapes the suite used to copy.

The server-action tests drove handlers against a send-log-only double,
the auth-gate tests needed handshake maps plus accept/close tracking,
and the protocol tests queued inbound frames. One double with optional
seed data serves all three, so a change to the fake socket is made in
one place instead of up to eighteen.
"""

from typing import Any, Dict, List, Optional

from fastapi import WebSocketDisconnect


class FakeWS:
    """Capture outbound frames; optionally queue inbound ones.

    ``incoming`` seeds frames ``receive_json`` yields before it drops
    the connection; ``query``/``headers``/``cookies``/``scope`` seed
    the handshake maps the auth gates read.
    """

    def __init__(
        self,
        incoming: Optional[List[Dict[str, Any]]] = None,
        query: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        scope: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Seed the queues and maps; track accept/close and outbound."""
        self._incoming = list(incoming or [])
        self.query_params: Dict[str, str] = query or {}
        self.headers: Dict[str, str] = headers or {}
        self.cookies: Dict[str, str] = cookies or {}
        self.scope: Dict[str, Any] = scope or {}
        self.accepted = False
        self.close_code: Optional[int] = None
        self.sent: List[Dict[str, Any]] = []

    async def accept(self) -> None:
        """Record that the handshake completed."""
        self.accepted = True

    async def close(self, code: int = 1000) -> None:
        """Record the close code the handler chose."""
        self.close_code = code

    async def receive_json(self) -> Dict[str, Any]:
        """Yield the next queued frame, or disconnect when drained."""
        if not self._incoming:
            raise WebSocketDisconnect()
        return self._incoming.pop(0)

    async def send_json(self, message: Dict[str, Any]) -> None:
        """Record one outbound message."""
        self.sent.append(message)
