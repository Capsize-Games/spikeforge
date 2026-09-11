"""Outbound WebSocket helpers for the model-hub actions.

Kept separate from :mod:`server.messages` so that module does not grow past
the project's file-length limit, exactly like :mod:`server.energy_messages`.
Each helper reuses the shared locked-send so the socket writes stay serialized
with every other reply.
"""

from typing import Any, Dict

from fastapi import WebSocket

from server.messages import send_locked
from server.session import Session


async def send_hub_list(
    ws: WebSocket, session: Session, payload: Dict[str, Any]
) -> None:
    """Send the annotated hub catalog to the client."""
    await send_locked(ws, session, {"type": "hub_list", "payload": payload})


async def send_hub_search(
    ws: WebSocket, session: Session, payload: Dict[str, Any]
) -> None:
    """Send hub search results with live-search availability."""
    await send_locked(ws, session, {"type": "hub_search", "payload": payload})


async def send_hub_download_state(
    ws: WebSocket, session: Session, state: Dict[str, Any]
) -> None:
    """Send a hub-download progress snapshot to the client."""
    await send_locked(ws, session, {
        "type": "hub_download_state", "payload": state,
    })


async def send_hub_inspect(
    ws: WebSocket, session: Session, payload: Dict[str, Any]
) -> None:
    """Send a hub artifact's structural report to the client."""
    await send_locked(ws, session, {"type": "hub_inspect", "payload": payload})


async def send_hub_import(
    ws: WebSocket, session: Session, payload: Dict[str, Any]
) -> None:
    """Send a hub import result, including any compatibility verdict."""
    await send_locked(ws, session, {"type": "hub_import", "payload": payload})
