"""Parity and contract tests for the refactored ``SpikingNet``."""

from pathlib import Path
from typing import Dict, Tuple

import torch

from spikeforge.network import inference
from spikeforge.network.spiking_net import SpikingNet

_LEGACY_KEYS = {"_fc1.weight", "_fc1.bias", "_fc2.weight", "_fc2.bias"}


def _lif_step(
    mem: torch.Tensor,
    current: torch.Tensor,
    beta: float,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One ``snn.Leaky`` step with reset-by-subtraction, as reference."""
    reset = (mem > threshold).float()
    mem = beta * mem + current
    spike = (mem > threshold).float()
    return mem - (spike - reset) * threshold, spike


def _reference(
    weights: Dict[str, torch.Tensor],
    spikes: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Hand-written fc1/lif1/fc2/lif2 recurrence for parity checks."""
    w1, b1 = weights["_fc1.weight"], weights["_fc1.bias"]
    w2, b2 = weights["_fc2.weight"], weights["_fc2.bias"]
    steps, batch = spikes.size(0), spikes.size(1)
    frames = spikes.reshape(steps, batch, -1)
    mem1 = torch.zeros(batch, w1.size(0))
    mem2 = torch.zeros(batch, w2.size(0))
    total = torch.zeros(batch, w2.size(0))
    for t in range(steps):
        mem1, spk1 = _lif_step(mem1, frames[t] @ w1.t() + b1, beta, 1.0)
        mem2, spk2 = _lif_step(mem2, spk1 @ w2.t() + b2, beta, 1.0)
        total = total + spk2
    return total / steps


def _net() -> SpikingNet:
    """Return a small deterministic network for the tests below."""
    return SpikingNet(hidden=8, beta=0.5, num_classes=4, input_size=6)


def test_forward_spikes_matches_reference_recurrence() -> None:
    """The refactored net reproduces the hand-written recurrence."""
    torch.manual_seed(0)
    net = _net()
    spikes = torch.rand(5, 3, 6)
    weights = {
        name: value
        for name, value in net.state_dict().items()
        if name in _LEGACY_KEYS
    }
    with torch.no_grad():
        got = net.forward_spikes(spikes)
        want = _reference(weights, spikes, net.beta)
    assert torch.allclose(got, want, atol=1e-6)


def test_checkpoint_state_dict_round_trip(tmp_path: Path) -> None:
    """A saved current-style state dict reloads with identical output."""
    torch.manual_seed(1)
    source = _net()
    state = source.state_dict()
    assert set(state) >= _LEGACY_KEYS
    path = tmp_path / "ckpt.pt"
    torch.save(state, path)
    target = _net()
    target.load_state_dict(torch.load(path, weights_only=True))
    spikes = torch.rand(4, 2, 6)
    with torch.no_grad():
        assert torch.equal(
            source.forward_spikes(spikes), target.forward_spikes(spikes)
        )


def test_track_contract_keys_and_shapes() -> None:
    """Tracking returns exactly the legacy keys and shaped traces."""
    net = _net()
    tracked = net.forward_spikes(torch.rand(5, 3, 6), track=True)
    assert set(tracked) == {"logits", "hidden", "output", "steps"}
    assert tracked["steps"] == 5
    assert len(tracked["hidden"]) == len(tracked["output"]) == 5
    assert tracked["hidden"][0].shape == (3, 8)
    assert tracked["output"][0].shape == (3, 4)


def test_membrane_key_is_additive() -> None:
    """``membrane=True`` adds a trace key without touching the rest."""
    net = _net()
    spikes = torch.rand(5, 3, 6)
    plain = net.forward_spikes(spikes, track=True)
    traced = net.forward_spikes(spikes, track=True, membrane=True)
    assert set(plain) <= set(traced)
    assert set(traced) == {
        "logits",
        "hidden",
        "output",
        "steps",
        "membrane",
    }
    assert traced["membrane"]["_lif1"][0].shape == (3, 8)
    assert torch.equal(plain["logits"], traced["logits"])


def test_forward_raw_pixels_matches_repeated_frames() -> None:
    """The legacy raw-pixel path equals a repeated flattened frame."""
    net = _net()
    images = torch.rand(3, 1, 2, 3)
    frames = images.view(3, -1).unsqueeze(0).repeat(4, 1, 1)
    with torch.no_grad():
        assert torch.equal(
            net.forward(images, 4), net.forward_spikes(frames)
        )


def test_infer_spikes_membrane_is_additive() -> None:
    """Inference gains a membrane key while keeping every old key."""
    net = _net()
    spikes = torch.rand(5, 1, 6)
    base = inference.infer_spikes(net, spikes, 4)
    traced = inference.infer_spikes(net, spikes, 4, membrane=True)
    assert set(base) <= set(traced)
    assert "membrane" not in base
    assert set(traced["membrane"]) == {"_lif1", "_lif2"}
    assert base["predicted"] == traced["predicted"]


def test_properties_are_preserved() -> None:
    """The public property accessors still report the constructor args."""
    net = SpikingNet(
        hidden=7, beta=0.25, num_classes=3, input_size=11
    )
    assert (net.hidden, net.beta, net.num_classes, net.input_size) == (
        7,
        0.25,
        3,
        11,
    )
