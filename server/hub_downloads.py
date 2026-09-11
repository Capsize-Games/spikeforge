"""Hub download wiring: the shared manager and its progress emitter.

Mirrors :mod:`server.downloads` for datasets. The manager runs the isolated
hub worker off the event loop, so a download never blocks the socket and can be
terminated to cancel. It keeps its own state type (``hub_download_state``); the
dataset ``download_state`` payload is untouched.
"""

from typing import Any, Dict

from fastapi import WebSocket

from server.hub_messages import send_hub_download_state
from server.session import Session
from spikeforge_hub.downloads import HubDownloadManager
from spikeforge_hub.errors import HubDownloadCancelledError

#: Hub download manager singleton, reached directly by a cancelling client.
manager = HubDownloadManager()


async def _emit(
    ws: WebSocket, session: Session, state: Dict[str, Any]
) -> None:
    """Forward one hub download-progress snapshot to the client."""
    await send_hub_download_state(ws, session, state)


async def ensure_hub_download(
    ws: WebSocket, session: Session, entry_id: str
) -> bool:
    """Download ``entry_id``; return False when the user cancelled it."""
    try:
        await manager.ensure(
            entry_id, lambda state: _emit(ws, session, state)
        )
    except HubDownloadCancelledError:
        return False
    return True
