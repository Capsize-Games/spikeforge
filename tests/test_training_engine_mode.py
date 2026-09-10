"""Execution-mode plumbing through the training engine."""

from typing import Any, Dict

import torch

from snn_interpreter.runtime.execution_mode import ExecutionMode
from snn_interpreter.simulator.runner import run
from snn_interpreter.training import training_engine as engine_mod
from snn_interpreter.training.training_engine import TrainingEngine


def _engine(**kwargs: Any) -> TrainingEngine:
    """Return a small download-free CPU engine."""
    defaults: Dict[str, Any] = {
        "dataset": "mnist", "num_steps": 4, "device": "cpu",
    }
    defaults.update(kwargs)
    return TrainingEngine(**defaults)


def test_default_mode_is_production() -> None:
    """The engine defaults to production, preserving prior behaviour."""
    engine = _engine()
    assert engine.mode == "production"
    assert engine.execution_mode is ExecutionMode.PRODUCTION


def test_educational_mode_maps_to_enum() -> None:
    """``mode="educational"`` selects the educational enum member."""
    engine = _engine(mode="educational")
    assert engine.mode == "educational"
    assert engine.execution_mode is ExecutionMode.EDUCATIONAL


def _spy_run(monkeypatch: Any, engine: TrainingEngine) -> Dict[str, Any]:
    """Patch ``run`` to record the mode it was forwarded."""
    seen: Dict[str, Any] = {}
    real = engine_mod.run

    def spy(
        net: Any, spikes: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        seen["mode"] = kwargs.get("mode")
        return real(net, spikes, *args, **kwargs)

    monkeypatch.setattr(engine_mod, "run", spy)
    engine._train_batch(torch.rand(4, 1, 28, 28), torch.randint(0, 10, (4,)))
    return seen


def test_train_batch_forwards_educational_mode(monkeypatch: Any) -> None:
    """A training step forwards the engine's educational mode to ``run``."""
    engine = _engine(hidden=4, beta=0.5, mode="educational")
    seen = _spy_run(monkeypatch, engine)
    assert seen["mode"] is ExecutionMode.EDUCATIONAL


def test_train_batch_forwards_production_default(monkeypatch: Any) -> None:
    """A default engine forwards production mode, unchanged from before."""
    engine = _engine(hidden=4, beta=0.5)
    seen = _spy_run(monkeypatch, engine)
    assert seen["mode"] is ExecutionMode.PRODUCTION


def test_production_records_nothing_and_educational_captures() -> None:
    """The selected mode drives trajectory capture on the same model."""
    engine = _engine(hidden=4, beta=0.5)
    spikes = engine._encode_batch(torch.rand(3, 1, 28, 28))
    production = run(engine.net, spikes, mode=ExecutionMode.PRODUCTION)
    educational = run(engine.net, spikes, mode=ExecutionMode.EDUCATIONAL)
    assert production.recorded is False
    assert educational.recorded is True
