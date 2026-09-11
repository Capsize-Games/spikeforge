"""AMP, gradient checkpointing, BPTT, and multi-GPU scale-ups."""

import math
from typing import Any, Dict, Optional, Tuple

import pytest
import torch

from server.schemas import TrainConfig
from server.training import TrainingService
from spikeforge.simulator.runner import run
from spikeforge.training.training_engine import TrainingEngine

_Grads = Dict[str, torch.Tensor]


def _engine(**kwargs: Any) -> TrainingEngine:
    """Return a CPU engine with small, download-free defaults."""
    defaults: Dict[str, Any] = {
        "dataset": "mnist", "num_steps": 4, "device": "cpu",
    }
    defaults.update(kwargs)
    return TrainingEngine(**defaults)


def _grads(
    engine: TrainingEngine, images: torch.Tensor, targets: torch.Tensor
) -> _Grads:
    """Run one optimisation step and return the resulting gradients."""
    engine._train_batch(images, targets)
    return {key: value.grad.detach().clone()
            for key, value in engine.net.named_parameters()
            if value.grad is not None}


def _backward(
    engine: TrainingEngine, spikes: torch.Tensor, bptt_steps: Optional[int]
) -> Tuple[torch.Tensor, _Grads]:
    """Return readout logits and grads for one summed-logit backward pass."""
    net = engine.net
    net.zero_grad(set_to_none=True)
    logits = run(net, spikes, bptt_steps=bptt_steps).logits
    logits.sum().backward()
    return logits, {key: value.grad.detach().clone()
                    for key, value in net.named_parameters()
                    if value.grad is not None}


def _recurrent(**kwargs: Any) -> TrainingEngine:
    """Return a small CPU recurrent engine for grad-path tests."""
    return _engine(topology="recurrent_net",
                   topology_params={"hidden": 9, "beta": 0.5}, **kwargs)


# --- B. mixed precision ------------------------------------------------


def test_amp_logits_close_to_fp32() -> None:
    """AMP autocast yields logits close to the fp32 path (guarded)."""
    torch.manual_seed(0)
    engine = _engine(hidden=8, beta=0.5, amp=True)
    if not engine.amp:
        pytest.skip("autocast unsupported on this device")
    assert engine.amp_dtype in ("bfloat16", "float16")
    spikes = engine._encode_batch(torch.rand(4, 1, 28, 28))
    with engine._amp.autocast():
        mixed = engine._logits(spikes)
    reference = engine._logits(spikes)
    assert torch.allclose(
        mixed.float(), reference.float(), atol=5e-2, rtol=1e-2
    )
    metrics = engine._train_batch(
        torch.rand(4, 1, 28, 28), torch.randint(0, 10, (4,))
    )
    assert math.isfinite(metrics["loss"])


# --- C. gradient checkpointing -----------------------------------------


def test_checkpointed_gradients_match_normal() -> None:
    """Checkpointed gradients match the non-checkpointed path."""
    images = torch.rand(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    plain = _engine(hidden=8, beta=0.5, seed=0)
    checked = _engine(hidden=8, beta=0.5, seed=0, grad_checkpoint=True)
    assert checked.grad_checkpoint is True
    left = _grads(plain, images, targets)
    right = _grads(checked, images, targets)
    assert left and set(left) == set(right)
    for key in left:
        assert torch.allclose(left[key], right[key], atol=1e-6)


# --- D. truncated BPTT --------------------------------------------------


def test_bptt_preserves_forward_and_truncates_gradients() -> None:
    """Truncated BPTT keeps forward values and changes the gradient tail."""
    torch.manual_seed(0)
    engine = _recurrent(num_steps=6)
    spikes = engine._encode_batch(torch.rand(4, 1, 28, 28))
    full, full_grads = _backward(engine, spikes, None)
    truncated, trunc_grads = _backward(engine, spikes, 2)
    assert torch.allclose(full, truncated, atol=1e-6)
    assert any(
        not torch.allclose(full_grads[key], trunc_grads[key], atol=1e-6)
        for key in full_grads
    )


def test_engine_bptt_flag_is_reported() -> None:
    """A configured BPTT window surfaces on the engine and trains."""
    engine = _recurrent(num_steps=6, bptt_steps=2)
    assert engine.bptt_steps == 2
    metrics = engine._train_batch(
        torch.rand(4, 1, 28, 28), torch.randint(0, 10, (4,))
    )
    assert math.isfinite(metrics["loss"])


# --- E. multi-GPU -------------------------------------------------------


def test_multi_gpu_degrades_cleanly_without_multiple_devices() -> None:
    """Requesting multi-GPU without several CUDA devices stays single."""
    engine = _engine(hidden=8, beta=0.5, multi_gpu=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        assert engine.multi_gpu is False
        assert engine.multi_gpu_status.startswith("unavailable")
    metrics = engine._train_batch(
        torch.rand(4, 1, 28, 28), torch.randint(0, 10, (4,))
    )
    assert math.isfinite(metrics["loss"])


# --- F. protocol wiring -------------------------------------------------


def test_train_config_scaleup_defaults() -> None:
    """The config's scale-up defaults are all off/unset."""
    config = TrainConfig(device="cpu")
    assert (config.amp, config.grad_checkpoint, config.multi_gpu) == (
        False, False, False
    )
    assert config.bptt_steps is None


def test_train_config_scaleup_flags_reach_engine() -> None:
    """Configured scale-up flags reach the built engine."""
    config = TrainConfig(
        device="cpu", topology="recurrent_net",
        topology_params={"hidden": 9}, amp=True, grad_checkpoint=True,
        bptt_steps=3, multi_gpu=True,
    )
    engine = TrainingService._make_engine(config, None, None)
    assert engine.grad_checkpoint is True
    assert engine.bptt_steps == 3
    assert engine.multi_gpu is False
    assert engine.multi_gpu_status.startswith("unavailable")
    assert isinstance(engine.amp, bool)
