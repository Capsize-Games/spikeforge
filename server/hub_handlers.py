"""Handlers for the model-hub WebSocket actions.

Mirrors :mod:`server.target_handlers`: it lives outside :mod:`server.handlers`
so that module stays within the project's 250-line limit, and ``dispatch``
reaches :func:`dispatch_hub` through the shared protocol router.

Every failure is reported on the existing ``error`` channel with a named
reason: an unknown entry, an unreadable artifact, or an incompatible import all
surface a message rather than a traceback or a silent partial load.
"""

from fastapi import WebSocket

from server.hub_downloads import ensure_hub_download, manager
from server.hub_messages import (
    send_hub_download_state,
    send_hub_import,
    send_hub_inspect,
    send_hub_list,
    send_hub_search,
)
from server.hub_payloads import (
    import_payload,
    inspect_payload,
    list_payload,
    search_payload,
)
from server.messages import send_locked
from server.schemas import ClientMessage
from server.session import Session
from snn_interpreter.hub.errors import HubError


async def _error(ws: WebSocket, session: Session, detail: str) -> None:
    """Report a hub failure on the existing error channel."""
    await send_locked(ws, session, {"type": "error", "payload": detail})


async def handle_hub_list(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Emit the filtered, availability-annotated hub catalog."""
    query = message.hub
    payload = list_payload(query.framework, query.kind, query.available)
    await send_hub_list(ws, session, payload)


async def handle_hub_search(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Emit catalog search results and live-search availability."""
    payload = search_payload(message.hub.query, message.hub.limit)
    await send_hub_search(ws, session, payload)


async def handle_hub_download(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Download an entry, streaming progress until it resolves."""
    entry_id = message.name or ""
    if not entry_id:
        await _error(ws, session, "hub_download requires a model name")
        return
    try:
        await ensure_hub_download(ws, session, entry_id)
    except HubError as error:
        await _error(ws, session, str(error))


async def handle_hub_cancel(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Terminate the in-flight download and report the terminal state."""
    manager.cancel()
    await send_hub_download_state(ws, session, manager.snapshot())


async def handle_hub_inspect(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Emit the structural report for an entry's artifact."""
    try:
        payload = inspect_payload(message.name or "")
    except HubError as error:
        await _error(ws, session, str(error))
        return
    await send_hub_inspect(ws, session, payload)


async def handle_hub_import(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Run the funnel and emit the verdict even when incompatible."""
    try:
        payload = import_payload(message.name or "", message.hub.topology)
    except HubError as error:
        await _error(ws, session, str(error))
        return
    await send_hub_import(ws, session, payload)


async def dispatch_hub(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route the hub list/search/download/cancel/inspect/import actions."""
    if message.type == "hub_list":
        await handle_hub_list(ws, session, message)
    elif message.type == "hub_search":
        await handle_hub_search(ws, session, message)
    elif message.type == "hub_download":
        await handle_hub_download(ws, session, message)
    elif message.type == "hub_cancel":
        await handle_hub_cancel(ws, session, message)
    elif message.type == "hub_inspect":
        await handle_hub_inspect(ws, session, message)
    else:
        await handle_hub_import(ws, session, message)
