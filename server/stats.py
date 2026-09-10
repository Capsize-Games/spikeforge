"""Resource-monitor statistics handler."""

from fastapi import WebSocket

from server.messages import send_locked
from server.session import Session
from snn_interpreter.runtime.system_stats import snapshot


async def handle_stats(ws: WebSocket, session: Session) -> None:
    """Send a CPU/GPU memory snapshot for the resource monitor."""
    await send_locked(ws, session, {
        "type": "system_stats",
        "payload": snapshot(session.training.engine),
    })
