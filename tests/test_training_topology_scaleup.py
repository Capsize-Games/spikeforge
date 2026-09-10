"""Conv and recurrent end-to-end training plus unchanged defaults."""

import math
from typing import Any, Dict

import pytest
import torch
from torch.nn.functional import cross_entropy

from server.schemas import EncodeConfig
from snn_interpreter.simulator.runner import run
from snn_interpreter.training.training_engine import TrainingEngine


def _engine(**kwargs: Any) -> TrainingEngine:
    """Return a CPU engine with small, download-free defaults."""
    defaults: Dict[str, Any] = {
        "dataset": "mnist", "num_steps": 4, "device": "cpu",
    }
    defaults.update(kwargs)
    return TrainingEngine(**defaults)


def _snapshot(engine: TrainingEngine) -> Dict[str, torch.Tensor]:
    """Return a copy of every tensor in the engine's state dict."""
    return {key: value.clone()
            for key, value in engine.net.state_dict().items()}


def _changed(before: Dict[str, torch.Tensor], engine: TrainingEngine) -> list:
    """Return the state-dict keys whose tensor changed after a step."""
    after = engine.net.state_dict()
    return [key for key in before if not torch.equal(before[key], after[key])]


# --- A. convolutional and recurrent training ---------------------------


def test_conv_net_trains_predicts_and_infers() -> None:
    """A conv topology completes steps, updates weights, and infers."""
    torch.manual_seed(0)
    engine = _engine(topology="conv_net", topology_params={"channels": 2},
                     hidden=8, beta=0.5)
    images = torch.rand(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    before = _snapshot(engine)
    losses = [engine._train_batch(images, targets)["loss"] for _ in range(3)]
    assert all(math.isfinite(loss) for loss in losses)
    assert _changed(before, engine)
    assert engine.predict(images).shape == (4,)
    assert engine.infer(torch.rand(4, 1, 784))["num_steps"] == 4


def test_recurrent_net_trains_predicts_and_infers() -> None:
    """A recurrent topology completes steps, updates weights, and infers."""
    torch.manual_seed(0)
    engine = _engine(topology="recurrent_net",
                     topology_params={"hidden": 9, "beta": 0.5}, num_steps=6)
    images = torch.rand(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    before = _snapshot(engine)
    losses = [engine._train_batch(images, targets)["loss"] for _ in range(3)]
    assert all(math.isfinite(loss) for loss in losses)
    assert _changed(before, engine)
    assert engine.predict(images).shape == (4,)
    assert engine.infer(torch.rand(6, 1, 784))["num_steps"] == 6


def test_conv_spatial_and_recurrent_feature_encoding() -> None:
    """Conv input reshapes spatially; recurrent input stays flat."""
    conv = _engine(topology="conv_net", topology_params={"channels": 2})
    rec = _engine(topology="recurrent_net",
                  topology_params={"hidden": 9}, num_steps=5)
    images = torch.rand(3, 1, 28, 28)
    assert conv._encode_batch(images).shape == (4, 3, 1, 28, 28)
    assert rec._encode_batch(images).shape == (5, 3, 28 * 28)


def test_encoded_spikes_reshape_for_conv_and_recurrent() -> None:
    """Rate-coded spikes reshape per topology input layout."""
    encode = EncodeConfig(coding="rate", num_steps=5)
    images = torch.rand(3, 1, 28, 28)
    conv = _engine(topology="conv_net", topology_params={"channels": 2},
                   encode=encode, num_steps=5)
    rec = _engine(topology="recurrent_net",
                  topology_params={"hidden": 9}, encode=encode, num_steps=5)
    assert conv._encode_batch(images).shape == (5, 3, 1, 28, 28)
    assert rec._encode_batch(images).shape == (5, 3, 28 * 28)


# --- defaults preserved ------------------------------------------------


def test_defaults_match_simulator_reference() -> None:
    """The default engine is numerically unchanged and reports off."""
    torch.manual_seed(0)
    engine = _engine(hidden=8, beta=0.5)
    images = torch.rand(6, 1, 28, 28)
    targets = torch.randint(0, 10, (6,))
    with torch.no_grad():
        logits = run(engine.net, engine._encode_batch(images)).logits
        expected = cross_entropy(logits, targets).item()
    metrics = engine._train_batch(images, targets)
    assert metrics["loss"] == pytest.approx(expected, abs=1e-6)
    assert (engine.amp, engine.amp_dtype) == (False, None)
    assert engine.grad_checkpoint is False
    assert engine.bptt_steps is None
    assert engine.multi_gpu is False


def test_run_scaleup_defaults_unchanged() -> None:
    """Passing the off values to ``run`` reproduces the default logits."""
    torch.manual_seed(0)
    engine = _engine(hidden=8, beta=0.5)
    spikes = engine._encode_batch(torch.rand(4, 1, 28, 28))
    base = run(engine.net, spikes).logits
    explicit = run(engine.net, spikes, grad_checkpoint=False,
                   bptt_steps=None).logits
    assert torch.equal(base, explicit)


def test_scale_up_status_block() -> None:
    """The status block exposes every flag for payloads."""
    status = _engine(hidden=8, beta=0.5).scale_up_status()
    assert status["amp"] is False
    assert status["amp_dtype"] is None
    assert status["grad_checkpoint"] is False
    assert status["bptt_steps"] is None
    assert status["multi_gpu"] is False
    assert status["multi_gpu_status"] == "disabled"
