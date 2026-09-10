"""Track one WebSocket session: its engine, stream task, and send lock."""

import asyncio
from typing import Optional


class Session:
    """Hold a client's encoder engine and cancellable stream task."""

    def __init__(self):
        self._engine = None
        self._task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()

    @property
    def engine(self):
        return self._engine

    def set_engine(self, engine):
        self._engine = engine

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, coro):
        """Cancel any running stream and start a new one."""
        await self.cancel()
        self._task = asyncio.create_task(coro)

    async def cancel(self):
        """Stop the active stream task, if one is running."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
