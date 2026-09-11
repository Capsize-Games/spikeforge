"""Handlers for the model-registry search and diff WebSocket actions.

Mirrors :mod:`server.target_handlers`: it lives outside :mod:`server.handlers`
so that module stays within the project's 250-line limit, and ``dispatch``
reaches :func:`dispatch_registry` through the shared protocol router. A bad
diff request is reported on the existing ``error`` channel rather than raising.
"""

from fastapi import WebSocket

from server.messages import send_locked, send_model_diff, send_model_search
from server.model_payloads import search_payload
from server.schemas import ClientMessage
from server.session import Session
from spikeforge.network import model_diff

#: Number of names a metadata diff requires.
DIFF_NAMES = 2


async def handle_model_search(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Emit the registry records matching the message query."""
    await send_model_search(ws, session, search_payload(message.query))


async def handle_model_diff(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Emit a metadata diff for the two named checkpoints."""
    names = list(message.query.names)
    if len(names) != DIFF_NAMES:
        await send_locked(ws, session, {
            "type": "error",
            "payload": "model_diff requires exactly two names",
        })
        return
    try:
        payload = model_diff.checkpoint_diff(names[0], names[1])
    except FileNotFoundError as error:
        await send_locked(ws, session, {
            "type": "error", "payload": str(error),
        })
        return
    await send_model_diff(ws, session, payload)


async def dispatch_registry(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route the model-registry search and diff actions."""
    if message.type == "model_search":
        await handle_model_search(ws, session, message)
    else:
        await handle_model_diff(ws, session, message)
