"""Run training in a worker thread and stream metrics to async callers."""

import asyncio
import threading
from typing import Optional

from snn_interpreter.training_engine import TrainingEngine


class TrainingService:
    """Bridge a blocking training loop into an asyncio-friendly queue."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._engine: Optional[TrainingEngine] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.queue: asyncio.Queue = asyncio.Queue()

    @property
    def engine(self):
        return self._engine

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, config):
        """Spawn a worker thread that trains and enqueues metrics."""
        self._stop.clear()
        self._engine = TrainingEngine(
            dataset=config.dataset,
            hidden=config.hidden,
            beta=config.beta,
            lr=config.lr,
            epochs=config.epochs,
            num_steps=config.num_steps,
            subset=config.subset,
            batch_size=config.batch_size,
            checkpoint=config.checkpoint,
        )
        self._thread = threading.Thread(
            target=self._run, args=(self._engine,), daemon=True
        )
        self._thread.start()

    def adopt(self, checkpoint, config):
        """Load a checkpoint into a ready-to-continue engine."""
        self._stop.clear()
        self._engine = TrainingEngine(
            dataset=config.dataset,
            hidden=config.hidden,
            beta=config.beta,
            lr=config.lr,
            epochs=config.epochs,
            num_steps=config.num_steps,
            subset=config.subset,
            batch_size=config.batch_size,
            checkpoint=checkpoint,
        )
        return self._engine

    def stop(self):
        """Signal the worker to stop after the current batch."""
        self._stop.set()

    def _run(self, engine: TrainingEngine):
        """Iterate metrics in the worker and push them to the queue."""
        try:
            for metrics in engine.train(should_stop=self._stop.is_set):
                self._put({"type": "train_metrics", "payload": metrics})
            self._put({"type": "train_state",
                       "payload": {"running": False, "reason": "finished"}})
        except Exception as exc:  # surface worker errors to the client
            self._put({"type": "error", "payload": str(exc)})
            self._put({"type": "train_state",
                       "payload": {"running": False, "reason": "error"}})
        finally:
            self._thread = None

    def _put(self, message):
        """Thread-safe enqueue onto the event loop."""
        self._loop.call_soon_threadsafe(self.queue.put_nowait, message)
