"""Outbound WebSocket helper for the event-driven energy action.

Kept separate from :mod:`server.messages` so that module does not grow past
the project's file-length limit; it reuses the shared locked-send helper so
the socket writes stay serialized exactly like every other reply.
"""

from typing import Any, Dict

from fastapi import WebSocket

from server.messages import send_locked
from server.session import Session


async def send_energy_report(
    ws: WebSocket, session: Session, payload: Dict[str, Any]
) -> None:
    """Send an event-driven energy/latency report to the client."""
    await send_locked(ws, session, {
        "type": "energy_report", "payload": payload,
    })
