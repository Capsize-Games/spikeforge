"""Hub download wiring: per-session managers and their progress emitters.

Mirrors :mod:`server.downloads` for datasets. Each session owns its
managers (see :class:`server.session.Session`), so a download never
blocks the socket, can be terminated to cancel, and never shares
progress or terminal state with another session. Managers keep their
own state type (``hub_download_state``); the dataset ``download_state``
payload is untouched. Community pulls get their own manager instance
with the same progress/cancel shape under ``hub_pull_state``, so a pull
never clobbers an in-flight curated download's snapshots.
"""

import asyncio
from typing import Any, Dict

from fastapi import WebSocket

from server.hub_messages import send_hub_download_state, send_hub_pull_state
from server.session import Session
from spikeforge_hub import community
from spikeforge_hub.catalog import get as catalog_get
from spikeforge_hub.errors import HubDownloadCancelledError, HubDownloadError


async def _emit(
    ws: WebSocket, session: Session, state: Dict[str, Any]
) -> None:
    """Forward one hub download-progress snapshot to the client."""
    await send_hub_download_state(ws, session, state)


async def _emit_pull(
    ws: WebSocket, session: Session, state: Dict[str, Any]
) -> None:
    """Forward one community-pull progress snapshot to the client."""
    await send_hub_pull_state(ws, session, state)


async def ensure_hub_download(
    ws: WebSocket, session: Session, entry_id: str
) -> bool:
    """Download ``entry_id``; return False when the user cancelled it."""
    try:
        await session.hub_downloads.ensure(
            entry_id, lambda state: _emit(ws, session, state)
        )
    except HubDownloadCancelledError:
        return False
    return True


async def ensure_hub_pull(
    ws: WebSocket, session: Session, entry_id: str
) -> bool:
    """Pull community ``entry_id``; return False when cancelled.

    The hosted index is refreshed first (off the event loop), so a pull
    always resolves against the latest published namespace; when the
    network is down the previously cached index covers for it. Only
    community ids are pulled here -- curated ids keep ``hub_download``.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, community.fetch_index)
    _require_community(entry_id)
    try:
        await session.hub_pulls.ensure(
            entry_id, lambda state: _emit_pull(ws, session, state)
        )
    except HubDownloadCancelledError:
        return False
    return True


def _require_community(entry_id: str) -> None:
    """Reject an unknown id, or a curated id sent to the pull action."""
    entry = catalog_get(entry_id)
    if entry is None:
        raise HubDownloadError(entry_id, "unknown catalog entry")
    if entry.source != "community":
        raise HubDownloadError(
            entry_id, "hub_pull needs a community id; use hub_download"
        )
