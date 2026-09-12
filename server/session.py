"""Track one WebSocket session: engine, stream task, and training."""

import asyncio
import contextlib
from typing import Any, Coroutine, Optional

from server.pipeline_service import PipelineService
from server.schemas import EncodeConfig
from server.training import TrainingService


class Session:
    """Hold a client's encoder engine, stream task, and trainer."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        """Create a session bound to the running event loop."""
        self._engine: Any = None
        self._encode_config: Optional[EncodeConfig] = None
        self._task: Optional[asyncio.Task] = None
        self._drain_task: Optional[asyncio.Task] = None
        self.training = TrainingService(loop)
        self.pipeline = PipelineService(loop)
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.lock = asyncio.Lock()

    @property
    def engine(self) -> Any:
        """Return the active encoder engine, if any."""
        return self._engine

    def set_engine(
        self, engine: Any, config: Optional[EncodeConfig] = None
    ) -> None:
        """Store the engine and, when given, its encoding config."""
        self._engine = engine
        if config is not None:
            self._encode_config = config

    def set_config(self, config: EncodeConfig) -> None:
        """Remember the latest EncodeConfig for payload context."""
        self._encode_config = config

    @property
    def encode_config(self) -> Optional[EncodeConfig]:
        """Return the most recent EncodeConfig, if any."""
        return self._encode_config

    @property
    def is_running(self) -> bool:
        """Return True while a stream task is active."""
        return self._task is not None and not self._task.done()

    async def start(self, coro: Coroutine[Any, Any, None]) -> None:
        """Cancel any running stream and start a new one."""
        await self.cancel()
        self._task = asyncio.create_task(coro)

    async def cancel(self) -> None:
        """Stop the active stream task, if one is running."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
