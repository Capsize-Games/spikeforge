"""Track one WebSocket session: engine, stream task, and training."""

import asyncio
import contextlib
import time
from collections import deque
from typing import Any, Coroutine, Deque, Optional

from server.downloads import DownloadManager
from server.pipeline_service import PipelineService
from server.schemas import ClientMessage, EncodeConfig
from server.training import TrainingService
from spikeforge_hub.downloads import HubDownloadManager

#: Max queued inbound messages per connection before the server drops.
INBOX_MAXSIZE = 64

#: Max messages accepted per sliding window before the server drops.
RATE_BURST = 120

#: Width of the sliding window the burst limit applies to, in seconds.
RATE_WINDOW_SECONDS = 1.0


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
        self.inbox: asyncio.Queue = asyncio.Queue(maxsize=INBOX_MAXSIZE)
        self._arrivals: Deque[float] = deque()
        self.lock = asyncio.Lock()
        # Each session owns its download managers, so one client's
        # cancel can never terminate another session's download.
        self.downloads = DownloadManager()
        self.hub_downloads = HubDownloadManager()
        self.hub_pulls = HubDownloadManager()

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

    def offer(self, message: ClientMessage) -> Optional[str]:
        """Queue one inbound message, or return why it was dropped.

        Returns None when the message was queued, ``"limited"`` when
        the connection exceeded its burst rate, and ``"full"`` when
        the inbox is full. Dropped messages never allocate beyond
        their parsed frame, so a flooding client cannot grow memory.
        """
        now = time.monotonic()
        arrivals = self._arrivals
        while arrivals and now - arrivals[0] > RATE_WINDOW_SECONDS:
            arrivals.popleft()
        if len(arrivals) >= RATE_BURST:
            return "limited"
        try:
            self.inbox.put_nowait(message)
        except asyncio.QueueFull:
            return "full"
        arrivals.append(now)
        return None

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
