"""Reproduce the classic STDP learning-window curve from real spike pairs."""

from dataclasses import dataclass
from typing import List, Tuple

import torch

from spikeforge.memory.stdp_synapse import STDPSynapse

#: Spike-pair offsets to sweep, in steps. Negative = post fires first.
DEFAULT_OFFSETS: Tuple[int, ...] = (-20, -10, -5, -2, -1, 1, 2, 5, 10, 20)


@dataclass
class StdpWindowResult:
    """One swept offset and the weight change it produced."""

    delta_t: int
    delta_w: float


def _pair_delta_w(delta_t: int, synapse_kwargs: dict) -> float:
    """Return the weight change from one isolated pre/post spike pair.

    ``delta_t`` is signed pre-minus-post: positive means pre fired
    ``delta_t`` steps before post (causal), negative means post fired
    first (anti-causal). The synapse starts at the range's midpoint so
    both potentiation and depression are visible, not clamped away.
    """
    synapse = STDPSynapse(1, 1, **synapse_kwargs)
    synapse.set_weight(torch.tensor([[0.0]]))
    pad = abs(delta_t) + 5
    steps = pad * 2
    pre_step = pad if delta_t >= 0 else pad + abs(delta_t)
    post_step = pad if delta_t < 0 else pad + abs(delta_t)
    before = float(synapse.weight.item())
    for t in range(steps):
        pre = torch.tensor([1.0]) if t == pre_step else torch.zeros(1)
        post = torch.tensor([1.0]) if t == post_step else torch.zeros(1)
        synapse.step(pre, post)
    return float(synapse.weight.item()) - before


def run_stdp_window_poc(
    offsets: Tuple[int, ...] = DEFAULT_OFFSETS,
    w_min: float = -1.0,
    w_max: float = 1.0,
) -> List[StdpWindowResult]:
    """Sweep isolated spike pairs and return the resulting learning window.

    This is issue #26-adjacent groundwork for on-chip/local learning: a
    genuinely timing-based rule (not a rate code, unlike
    :class:`~spikeforge.memory.hebbian_synapse.HebbianSynapse`), verified
    by its shape rather than asserted -- causal pairs must potentiate,
    anti-causal pairs must depress, and the effect must shrink as the
    pair's spikes move further apart.
    """
    kwargs = {"w_min": w_min, "w_max": w_max}
    return [
        StdpWindowResult(delta_t=dt, delta_w=_pair_delta_w(dt, kwargs))
        for dt in offsets
    ]
