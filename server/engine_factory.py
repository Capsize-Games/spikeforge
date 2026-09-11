"""Build encoder engines, ensuring the dataset is present first."""

import asyncio
from typing import Any, Dict, Type

from fastapi import WebSocket

from server.download_errors import DownloadCancelledError
from server.downloads import manager
from server.encoder import EncoderEngine
from server.event_engine import EventEngine
from server.messages import send_download_state
from server.schemas import EncodeConfig
from server.session import Session
from spikeforge.data.datasets import (
    dataset_available,
    dataset_modality,
)


async def _emit(
    ws: WebSocket, session: Session, state: Dict[str, Any]
) -> None:
    """Forward one download-progress snapshot to the client."""
    await send_download_state(ws, session, state)


async def ensure_dataset(
    ws: WebSocket, session: Session, dataset: str, train: bool = True
) -> bool:
    """Ensure a dataset is on disk; return False when cancelled.

    An event dataset whose ``tonic`` loader is absent serves its explicit
    synthetic path, so there is nothing to fetch and the download worker is
    skipped instead of failing.
    """
    if not _needs_download(dataset):
        return True
    try:
        await manager.ensure(
            dataset, train, lambda state: _emit(ws, session, state)
        )
    except DownloadCancelledError:
        return False
    return True


def _needs_download(dataset: str) -> bool:
    """Return False for event datasets whose loader is absent.

    A missing ``tonic`` extra means the event source falls back to its
    explicit synthetic path, so there is nothing to fetch and spawning the
    download worker would only fail.
    """
    offline = (
        dataset_modality(dataset) == "event"
        and not dataset_available(dataset)
    )
    return not offline


def _engine_class(dataset: str) -> Type[Any]:
    """Return the engine class the dataset's modality selects."""
    if dataset_modality(dataset) == "event":
        return EventEngine
    return EncoderEngine


async def build_encoder(
    ws: WebSocket, session: Session, cfg: EncodeConfig
) -> bool:
    """Ensure the dataset, then build and store the matching encoder engine."""
    if not await ensure_dataset(ws, session, cfg.dataset):
        return False
    loop = asyncio.get_running_loop()
    factory = _engine_class(cfg.dataset)
    engine = await loop.run_in_executor(None, factory, cfg)
    session.set_engine(engine, cfg)
    return True
