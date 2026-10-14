"""Handlers for the model-hub WebSocket actions.

Mirrors :mod:`server.target_handlers`: it lives outside :mod:`server.handlers`
so that module stays within the project's 250-line limit, and ``dispatch``
reaches :func:`dispatch_hub` through the shared protocol router.

Every failure is reported on the existing ``error`` channel with a named
reason: an unknown entry, an unreadable artifact, or an incompatible import all
surface a message rather than a traceback or a silent partial load.
"""

from fastapi import WebSocket

from server.hub_downloads import ensure_hub_download, ensure_hub_pull
from server.hub_messages import (
    send_bundle_built,
    send_hub_download_state,
    send_hub_import,
    send_hub_inspect,
    send_hub_list,
    send_hub_pull_state,
    send_hub_search,
)
from server.hub_payloads import (
    bundle_payload,
    import_payload,
    inspect_payload,
    list_payload,
    search_payload,
)
from server.messages import reject_if_read_only, send_locked
from server.schemas import ClientMessage
from server.session import Session
from spikeforge_hub.errors import HubError


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
    if await reject_if_read_only(
        ws, session, "hub downloads are disabled on the public demo"
    ):
        return
    entry_id = message.name or ""
    if not entry_id:
        await _error(ws, session, "hub_download requires a model name")
        return
    try:
        await ensure_hub_download(ws, session, entry_id)
    except HubError as error:
        await _error(ws, session, str(error))


async def cancel_hub_activity(ws: WebSocket, session: Session) -> None:
    """Terminate the session's in-flight hub download or pull.

    Shared by the dispatched ``hub_cancel`` branch and the socket fast
    path in :mod:`server.app`, so a pull cancelled over a real socket is
    terminated exactly like one cancelled through ``dispatch``.
    """
    pull_active = (
        session.hub_pulls.snapshot()["status"] == "downloading"
    )
    session.hub_downloads.cancel()
    session.hub_pulls.cancel()
    await send_hub_download_state(
        ws, session, session.hub_downloads.snapshot()
    )
    if pull_active:
        await send_hub_pull_state(
            ws, session, session.hub_pulls.snapshot()
        )


async def handle_hub_cancel(
    ws: WebSocket, session: Session, _message: ClientMessage
) -> None:
    """Terminate the in-flight download or pull and report its state."""
    await cancel_hub_activity(ws, session)


async def handle_hub_inspect(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Emit the structural report for an entry's artifact."""
    if await reject_if_read_only(
        ws, session, "hub inspection is disabled on the public demo"
    ):
        return
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
    if await reject_if_read_only(
        ws, session, "hub imports are disabled on the public demo"
    ):
        return
    try:
        payload = import_payload(message.name or "", message.hub.topology)
    except HubError as error:
        await _error(ws, session, str(error))
        return
    await send_hub_import(ws, session, payload)


async def handle_hub_pull(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Pull a community entry, streaming progress until it resolves."""
    if await reject_if_read_only(
        ws, session, "hub downloads are disabled on the public demo"
    ):
        return
    entry_id = message.name or ""
    if not entry_id:
        await _error(ws, session, "hub_pull requires a model name")
        return
    try:
        await ensure_hub_pull(ws, session, entry_id)
    except HubError as error:
        await _error(ws, session, str(error))


async def handle_bundle_build(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Build a saved model into a ``.spkf`` and report its identity."""
    name = message.name or ""
    if not name:
        await _error(ws, session, "bundle_build requires a model name")
        return
    try:
        payload = bundle_payload(name)
    except HubError as error:
        await _error(ws, session, str(error))
        return
    await send_bundle_built(ws, session, payload)


async def dispatch_hub(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route the hub and bundle-build actions."""
    if message.type == "hub_list":
        await handle_hub_list(ws, session, message)
    elif message.type == "hub_search":
        await handle_hub_search(ws, session, message)
    elif message.type == "hub_download":
        await handle_hub_download(ws, session, message)
    elif message.type == "hub_cancel":
        await handle_hub_cancel(ws, session, message)
    elif message.type == "hub_pull":
        await handle_hub_pull(ws, session, message)
    elif message.type == "bundle_build":
        await handle_bundle_build(ws, session, message)
    elif message.type == "hub_inspect":
        await handle_hub_inspect(ws, session, message)
    else:
        await handle_hub_import(ws, session, message)
