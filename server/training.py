"""Run training in a worker thread and stream metrics to async callers."""

import asyncio
import threading
import time
from typing import Optional

from snn_interpreter import model_store
from snn_interpreter.training_engine import TrainingEngine


def _checkpoint_meta(checkpoint):
    """Return a checkpoint's stored meta, or an empty dict."""
    if not checkpoint:
        return {}
    try:
        return model_store.load(checkpoint).get("meta", {})
    except Exception:
        return {}


def _dataset_from(config, encode):
    """Prefer the encode config's dataset, else the train config's."""
    return encode.dataset if encode is not None else config.dataset


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
    def input_mode(self):
        """Return the active engine's input mode, defaulting to raw."""
        if self._engine is None:
            return "raw"
        return self._engine.input_mode

    def infer(self, spikes, true_label=None):
        """Score encoded spikes with the active engine, if any."""
        if self._engine is None:
            return None
        return self._engine.infer(spikes, true_label)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, config, encode=None):
        """Spawn a worker thread that trains and enqueues metrics."""
        self._stop.clear()
        self._engine = self._make_engine(config, encode, config.checkpoint)
        self._thread = threading.Thread(
            target=self._run, args=(self._engine,), daemon=True
        )
        self._thread.start()

    def adopt(self, checkpoint, config, encode=None):
        """Load a checkpoint into a ready-to-continue engine."""
        self._stop.clear()
        self._engine = self._make_engine(config, encode, checkpoint)
        return self._engine

    @staticmethod
    def _make_engine(config, encode, checkpoint):
        """Build a TrainingEngine, honoring a checkpoint's architecture."""
        meta = _checkpoint_meta(checkpoint)
        dataset = meta.get("dataset") or _dataset_from(config, encode)
        return TrainingEngine(
            dataset=dataset,
            hidden=int(meta.get("hidden", config.hidden)),
            beta=float(meta.get("beta", config.beta)),
            lr=config.lr,
            epochs=config.epochs,
            num_steps=config.num_steps,
            subset=config.subset,
            batch_size=config.batch_size,
            checkpoint=checkpoint,
            encode=encode,
            device=config.device,
        )

    def stop(self):
        """Signal the worker to stop after the current batch."""
        self._stop.set()

    def _run(self, engine: TrainingEngine):
        """Iterate metrics in the worker and push them to the queue."""
        try:
            last = time.perf_counter()
            for metrics in engine.train(should_stop=self._stop.is_set):
                now = time.perf_counter()
                metrics["device"] = engine.device
                metrics["step_ms"] = round((now - last) * 1000.0, 1)
                last = now
                self._put({"type": "train_metrics", "payload": metrics})
            self._put({"type": "train_state",
                       "payload": {"running": False, "reason": "finished",
                                   "device": engine.device}})
        except Exception as exc:  # surface worker errors to the client
            self._put({"type": "error", "payload": str(exc)})
            self._put({"type": "train_state",
                       "payload": {"running": False, "reason": "error",
                                   "device": engine.device}})
        finally:
            self._thread = None

    def _put(self, message):
        """Thread-safe enqueue onto the event loop."""
        self._loop.call_soon_threadsafe(self.queue.put_nowait, message)
