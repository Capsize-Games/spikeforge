"""Build encoder engines, ensuring the dataset is present first."""

import asyncio
from typing import Any, Dict

from fastapi import WebSocket

from server.download_errors import DownloadCancelledError
from server.downloads import manager
from server.encoder import EncoderEngine
from server.messages import send_download_state
from server.schemas import EncodeConfig
from server.session import Session


async def _emit(
    ws: WebSocket, session: Session, state: Dict[str, Any]
) -> None:
    """Forward one download-progress snapshot to the client."""
    await send_download_state(ws, session, state)


async def ensure_dataset(
    ws: WebSocket, session: Session, dataset: str, train: bool = True
) -> bool:
    """Ensure a dataset is on disk; return False when cancelled."""
    try:
        await manager.ensure(
            dataset, train, lambda state: _emit(ws, session, state)
        )
    except DownloadCancelledError:
        return False
    return True


async def build_encoder(
    ws: WebSocket, session: Session, cfg: EncodeConfig
) -> bool:
    """Ensure the dataset, then build and store the encoder engine."""
    if not await ensure_dataset(ws, session, cfg.dataset):
        return False
    loop = asyncio.get_running_loop()
    engine = await loop.run_in_executor(None, EncoderEngine, cfg)
    session.set_engine(engine, cfg)
    return True
