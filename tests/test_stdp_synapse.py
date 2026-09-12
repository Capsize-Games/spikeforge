"""Verify STDPSynapse learns from real spike timing, not spike rate."""

import torch

from spikeforge.memory.stdp_synapse import STDPSynapse


def _pair(delta_t: int, w_min: float = -1.0, w_max: float = 1.0) -> float:
    """Run one pre/post spike pair separated by ``delta_t`` steps.

    Positive ``delta_t`` fires pre before post (causal); negative fires
    post before pre (anti-causal). Returns the resulting change in the
    single synapse's weight from its 0.5 starting point.
    """
    synapse = STDPSynapse(1, 1, w_min=w_min, w_max=w_max)
    synapse.set_weight(torch.tensor([[0.5]]))
    steps = abs(delta_t) + 10
    pre_step = 2 if delta_t >= 0 else 2 + abs(delta_t)
    post_step = 2 if delta_t < 0 else 2 + abs(delta_t)
    for t in range(steps):
        pre = torch.tensor([1.0]) if t == pre_step else torch.zeros(1)
        post = torch.tensor([1.0]) if t == post_step else torch.zeros(1)
        synapse.step(pre, post)
    return float(synapse.weight.item()) - 0.5


def test_causal_pairing_potentiates() -> None:
    """Pre-before-post strengthens the synapse."""
    assert _pair(delta_t=5) > 0


def test_anticausal_pairing_depresses() -> None:
    """Post-before-pre weakens the synapse."""
    assert _pair(delta_t=-5) < 0


def test_closer_pairing_has_larger_magnitude() -> None:
    """The exponential trace makes a tighter pairing count for more.

    This is the defining shape of the STDP learning window: the same
    causal pairing at a shorter latency produces a bigger weight change
    because less of the eligibility trace has decayed away.
    """
    close = _pair(delta_t=2)
    far = _pair(delta_t=15)
    assert close > far > 0


def test_no_spikes_no_change() -> None:
    """With no spikes at all, the weight never moves."""
    synapse = STDPSynapse(3, 2)
    synapse.set_weight(torch.full((3, 2), 0.5))
    for _ in range(20):
        synapse.step(torch.zeros(3), torch.zeros(2))
    assert torch.equal(synapse.weight, torch.full((3, 2), 0.5))


def test_weight_clamped_to_bounds() -> None:
    """Repeated causal pairing saturates at w_max, never overshoots."""
    synapse = STDPSynapse(1, 1, w_max=0.6, a_plus=0.5)
    synapse.set_weight(torch.tensor([[0.5]]))
    for _ in range(50):
        synapse.step(torch.tensor([1.0]), torch.zeros(1))
        synapse.step(torch.zeros(1), torch.tensor([1.0]))
    assert synapse.weight.item() <= 0.6 + 1e-6
