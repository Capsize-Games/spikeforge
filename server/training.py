"""Run training in a worker thread and stream metrics to async callers."""

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional, Type

import torch

from server.schemas import EncodeConfig, TrainConfig
from spikeforge.data.datasets import dataset_modality
from spikeforge.network import model_store
from spikeforge.observability import persistence
from spikeforge.training.event_engine import EventTrainingEngine
from spikeforge.training.training_engine import TrainingEngine


def _checkpoint_meta(checkpoint: Optional[str]) -> Dict[str, Any]:
    """Return a checkpoint's stored meta, or an empty dict."""
    if not checkpoint:
        return {}
    try:
        return model_store.load(checkpoint).get("meta", {})
    except Exception:
        return {}


def _dataset_from(config: TrainConfig, encode: Optional[EncodeConfig]) -> str:
    """Prefer the encode config's dataset, else the train config's."""
    return encode.dataset if encode is not None else config.dataset


def _topology_params(
    config: TrainConfig, meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Return topology params with per-stage neuron overrides folded in.

    A checkpoint's stored params win (they reproduce the trained
    architecture), and the config's additive ``stage_neurons``/
    ``stage_params`` are layered on top so heterogeneous neurons survive the
    config-to-engine hand-off.
    """
    params = dict(meta.get("topology_params") or config.topology_params)
    if config.stage_neurons:
        params["neurons"] = dict(config.stage_neurons)
    if config.stage_params:
        params["stage_params"] = dict(config.stage_params)
    return params


def _engine_class(dataset: str) -> Type[Any]:
    """Return the training engine class the dataset's modality selects.

    Image modality keeps the historical :class:`TrainingEngine`; event
    modality builds the event-stream engine so one server route trains both.
    """
    if dataset_modality(dataset) == "event":
        return EventTrainingEngine
    return TrainingEngine


def _point(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Trim a metrics dict to the history fields we persist."""
    return {
        "step": metrics["step"],
        "epoch": metrics["epoch"],
        "loss": metrics["loss"],
        "train_accuracy": metrics["train_accuracy"],
        "test_accuracy": metrics["test_accuracy"],
    }


class TrainingService:
    """Bridge a blocking training loop into an asyncio-friendly queue."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the service to the loop that owns the send queue."""
        self._loop = loop
        self._engine: Optional[TrainingEngine] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._history: List[Dict[str, Any]] = []
        self.queue: asyncio.Queue = asyncio.Queue()

    @property
    def engine(self) -> Optional[TrainingEngine]:
        """Return the active training engine, if any."""
        return self._engine

    @property
    def input_mode(self) -> str:
        """Return the active engine's input mode, defaulting to raw."""
        if self._engine is None:
            return "raw"
        return self._engine.input_mode

    @property
    def mode(self) -> str:
        """Return the active engine's execution mode, defaulting production."""
        if self._engine is None:
            return "production"
        return self._engine.mode

    def infer(
        self, spikes: torch.Tensor, true_label: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Score encoded spikes with the active engine, if any."""
        if self._engine is None:
            return None
        return self._engine.infer(spikes, true_label)

    @property
    def is_running(self) -> bool:
        """Return True while the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def clear(self) -> None:
        """Stop training and drop the active engine."""
        self.stop()
        self._engine = None

    def start(
        self, config: TrainConfig, encode: Optional[EncodeConfig] = None
    ) -> None:
        """Spawn a worker that builds the engine, then trains."""
        self._stop.clear()
        self._history = []
        self._thread = threading.Thread(
            target=self._build_and_run, args=(config, encode), daemon=True
        )
        self._thread.start()

    def _build_and_run(
        self, config: TrainConfig, encode: Optional[EncodeConfig]
    ) -> None:
        """Build the engine off the event loop, then stream metrics."""
        try:
            engine = self._make_engine(config, encode, config.checkpoint)
        except Exception as exc:  # surface build failures to the client
            self._put({"type": "error", "payload": str(exc)})
            self._put({"type": "train_state",
                       "payload": {"running": False, "reason": "error",
                                   "mode": config.mode}})
            self._thread = None
            return
        self._engine = engine
        self._run(engine)

    def adopt(
        self,
        checkpoint: str,
        config: TrainConfig,
        encode: Optional[EncodeConfig] = None,
    ) -> TrainingEngine:
        """Load a checkpoint, restoring its metric history too."""
        self._stop.clear()
        self._engine = self._make_engine(config, encode, checkpoint)
        history = model_store.load(checkpoint).get("history", [])
        self._history = history if _checkpoint_meta(checkpoint) else []
        return self._engine

    def save(self, name: str) -> str:
        """Persist the active model together with its metric history."""
        return self._engine.save(name, self._history)

    @staticmethod
    def _make_engine(config: TrainConfig, encode: Optional[EncodeConfig],
                     checkpoint: Optional[str]) -> TrainingEngine:
        """Build the modality-appropriate engine for the dataset."""
        meta = _checkpoint_meta(checkpoint)
        dataset = meta.get("dataset") or _dataset_from(config, encode)
        params = _topology_params(config, meta)
        engine_class = _engine_class(dataset)
        return engine_class(
            dataset=dataset, hidden=int(meta.get("hidden", config.hidden)),
            beta=float(meta.get("beta", config.beta)), lr=config.lr,
            epochs=config.epochs, num_steps=config.num_steps,
            subset=config.subset, batch_size=config.batch_size,
            checkpoint=checkpoint, encode=encode, device=config.device,
            topology=str(meta.get("topology", config.topology)),
            topology_params=dict(params), mode=config.mode,
            amp=config.amp, grad_checkpoint=config.grad_checkpoint,
            bptt_steps=config.bptt_steps, multi_gpu=config.multi_gpu,
            tracking=config.tracking, deterministic=config.deterministic,
        )

    def stop(self) -> None:
        """Signal the worker to stop after the current batch."""
        self._stop.set()

    def _run(self, engine: TrainingEngine) -> None:
        """Iterate metrics in the worker and push them to the queue."""
        try:
            last = time.perf_counter()
            for metrics in engine.train(should_stop=self._stop.is_set):
                elapsed = (time.perf_counter() - last) * 1000.0
                metrics["step_ms"] = round(elapsed, 1)
                last = time.perf_counter()
                metrics["device"] = engine.device
                metrics["scaleup"] = engine.scale_up_status()
                self._history.append(_point(metrics))
                self._put({"type": "train_metrics", "payload": metrics})
            self._state("finished", engine)
        except Exception as exc:  # surface worker errors to the client
            self._put({"type": "error", "payload": str(exc)})
            self._state("error", engine)
        finally:
            # Durability point: persist metrics at the end of a run when the
            # opt-in flag is set; a no-op otherwise, so the default is safe.
            persistence.flush()
            self._thread = None

    def _state(self, reason: str, engine: TrainingEngine) -> None:
        """Queue a stopped train_state carrying device, mode, and scaleups."""
        self._put({"type": "train_state",
                   "payload": {"running": False, "reason": reason,
                               "device": engine.device,
                               "mode": engine.mode,
                               "scaleup": engine.scale_up_status()}})

    def _put(self, message: Dict[str, Any]) -> None:
        """Thread-safe enqueue onto the event loop."""
        self._loop.call_soon_threadsafe(self.queue.put_nowait, message)
