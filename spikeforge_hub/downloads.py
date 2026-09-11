"""Hub downloads: isolated worker process, progress, verification, and cancel.

Mirrors :class:`server.downloads.DownloadManager` so a hub download never
blocks the FastAPI event loop and can be terminated to cancel in flight. It
keeps its own state type (``hub_download_state``) with ``verified`` added, and
its terminal statuses stay ``{idle, downloading, done, cancelled, error}`` for
consistency with the dataset downloader.
"""

import asyncio
import json
import subprocess
import sys
from typing import Any, Awaitable, Callable, Dict, Optional

from spikeforge_hub import cache
from spikeforge_hub.catalog import get as catalog_get
from spikeforge_hub.errors import (
    HubDownloadCancelledError,
    HubDownloadError,
)

_POLL_SECONDS = 0.4

Emit = Callable[[Dict[str, Any]], Awaitable[None]]


def _spawn(entry_id: str) -> subprocess.Popen:
    """Start the isolated hub downloader child process."""
    return subprocess.Popen(
        [sys.executable, "-m", "spikeforge_hub.download_cli", entry_id],
        stdout=subprocess.PIPE,
        text=True,
    )


def _report(process: Optional[subprocess.Popen]) -> Dict[str, Any]:
    """Parse the last JSON line the worker printed (best effort)."""
    if process is None or process.stdout is None:
        return {}
    text = process.stdout.read() or ""
    for line in reversed(text.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def _verified(process: Optional[subprocess.Popen]) -> bool:
    """Return the worker's ``verified`` flag, defaulting to False."""
    return bool(_report(process).get("verified", False))


class HubDownloadManager:
    """Download hub entries off the event loop and stream progress."""

    def __init__(self) -> None:
        """Start idle with no entry selected."""
        self._process: Optional[subprocess.Popen] = None
        self._entry_id = ""
        self._status = "idle"
        self._bytes = 0
        self._baseline = 0
        self._total: Optional[int] = None
        self._verified = False

    def snapshot(self) -> Dict[str, Any]:
        """Return the current hub download state for the client."""
        return {
            "id": self._entry_id,
            "status": self._status,
            "bytes": self._bytes,
            "total_bytes": self._total,
            "verified": self._verified,
        }

    def cancel(self) -> None:
        """Terminate the worker process if a download is in flight."""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._status = "cancelled"

    async def ensure(self, entry_id: str, emit: Emit) -> None:
        """Download ``entry_id`` into the cache, emitting progress."""
        entry = catalog_get(entry_id)
        if entry is None:
            raise HubDownloadError(entry_id, "unknown catalog entry")
        loop = asyncio.get_running_loop()
        await self._download(loop, entry_id, entry.size_bytes, emit)

    async def _download(
        self,
        loop: asyncio.AbstractEventLoop,
        entry_id: str,
        total: Optional[int],
        emit: Emit,
    ) -> None:
        """Run the worker process, polling progress until it exits."""
        self._entry_id = entry_id
        self._status = "downloading"
        self._total = total
        self._verified = False
        self._baseline = cache.entry_bytes(entry_id)
        self._bytes = 0
        await emit(self.snapshot())
        self._process = _spawn(entry_id)
        await self._poll(loop, entry_id, emit)
        await self._finish(emit)

    async def _poll(
        self,
        loop: asyncio.AbstractEventLoop,
        entry_id: str,
        emit: Emit,
    ) -> None:
        """Emit progress snapshots until the worker process exits."""
        while self._process is not None and self._process.poll() is None:
            current = await loop.run_in_executor(
                None, cache.entry_bytes, entry_id
            )
            self._bytes = max(0, current - self._baseline)
            await emit(self.snapshot())
            await asyncio.sleep(_POLL_SECONDS)

    async def _finish(self, emit: Emit) -> None:
        """Resolve the terminal state and report it to the client."""
        process = self._process
        self._process = None
        if self._status == "cancelled":
            await emit(self.snapshot())
            raise HubDownloadCancelledError(self._entry_id)
        if process is None or process.returncode != 0:
            self._status = "error"
            await emit(self.snapshot())
            raise HubDownloadError(self._entry_id, "hub download failed")
        self._verified = _verified(process)
        self._status = "done"
        await emit(self.snapshot())


manager = HubDownloadManager()
