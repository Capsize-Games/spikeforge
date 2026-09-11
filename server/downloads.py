"""Dataset downloads: isolated worker process, progress, and cancellation."""

import asyncio
import os
import subprocess
import sys
from typing import Any, Awaitable, Callable, Dict, Optional, Set

from server.download_errors import DownloadCancelledError
from spikeforge.config import DATA_DIR
from spikeforge.data.datasets import build_dataset

_POLL_SECONDS = 0.4

Emit = Callable[[Dict[str, Any]], Awaitable[None]]


def _spawn(dataset: str, train: bool) -> subprocess.Popen:
    """Start the isolated downloader child process."""
    return subprocess.Popen([
        sys.executable,
        "-m",
        "spikeforge.data.download_cli",
        dataset,
        "1" if train else "0",
    ])


def _dir_bytes(path: str) -> int:
    """Sum the sizes of every file under ``path`` (best effort)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


class DownloadManager:
    """Prepare datasets off the event loop and stream progress to a client."""

    def __init__(self) -> None:
        """Start idle, with an empty set of already-prepared datasets."""
        self._prepared: Set[str] = set()
        self._process: Optional[subprocess.Popen] = None
        self._dataset = ""
        self._status = "idle"
        self._bytes = 0
        self._baseline = 0

    def snapshot(self) -> Dict[str, Any]:
        """Return the current download state for the client."""
        return {
            "dataset": self._dataset,
            "status": self._status,
            "bytes": self._bytes,
        }

    def cancel(self) -> None:
        """Terminate the worker process if a download is in flight."""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._status = "cancelled"

    async def ensure(self, dataset: str, train: bool, emit: Emit) -> None:
        """Ensure ``dataset`` is on disk, emitting progress until it is."""
        if dataset in self._prepared:
            return
        loop = asyncio.get_running_loop()
        if await loop.run_in_executor(None, self._loads, dataset, train):
            self._prepared.add(dataset)
            return
        await self._download(loop, dataset, train, emit)
        self._prepared.add(dataset)

    @staticmethod
    def _loads(dataset: str, train: bool) -> bool:
        """Return True when the dataset is already available locally."""
        try:
            build_dataset(dataset, train=train, download=False)
            return True
        except (RuntimeError, OSError, ValueError, KeyError):
            return False

    async def _download(
        self,
        loop: asyncio.AbstractEventLoop,
        dataset: str,
        train: bool,
        emit: Emit,
    ) -> None:
        """Run the worker process, polling progress until it exits."""
        self._dataset = dataset
        self._status = "downloading"
        self._baseline = _dir_bytes(DATA_DIR)
        self._bytes = 0
        await emit(self.snapshot())
        self._process = _spawn(dataset, train)
        await self._poll(loop, emit)
        await self._finish(emit)

    async def _poll(
        self, loop: asyncio.AbstractEventLoop, emit: Emit
    ) -> None:
        """Emit progress snapshots until the worker process exits."""
        while self._process is not None and self._process.poll() is None:
            current = await loop.run_in_executor(None, _dir_bytes, DATA_DIR)
            self._bytes = max(0, current - self._baseline)
            await emit(self.snapshot())
            await asyncio.sleep(_POLL_SECONDS)

    async def _finish(self, emit: Emit) -> None:
        """Resolve the terminal state and report it to the client."""
        process = self._process
        self._process = None
        if self._status == "cancelled":
            await emit(self.snapshot())
            raise DownloadCancelledError()
        if process is None or process.returncode != 0:
            self._status = "error"
            await emit(self.snapshot())
            raise RuntimeError("dataset download failed")
        self._status = "done"
        await emit(self.snapshot())


manager = DownloadManager()
