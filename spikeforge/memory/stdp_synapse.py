"""A feedforward synapse written by pair-based spike-timing plasticity."""

from typing import Optional

import torch


class STDPSynapse:
    """Weights written step-by-step from real pre/post spike timing.

    :class:`~spikeforge.memory.hebbian_synapse.HebbianSynapse` writes one
    column in a single outer-product shot from a summed spike-count trace,
    which is a rate code, not a timing code. This synapse instead tracks
    exponentially-decaying eligibility traces for both sides of the
    connection and updates every weight at every timestep from the
    genuine order of spikes: a postsynaptic spike potentiates whichever
    presynaptic inputs recently fired (pre-before-post, causal), and a
    presynaptic spike depresses whichever outputs recently fired
    (post-before-pre, anti-causal). No optimizer, no loss, no backward
    pass touches these weights — only :meth:`step`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        w_min: float = 0.0,
        w_max: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """Start from zero weights and zero eligibility traces.

        ``a_minus`` slightly exceeds ``a_plus`` (the classic Song/Abbott
        ratio) so an uncorrelated pre/post pair decays rather than drifts
        toward saturation; ``tau_pre``/``tau_post`` set how many steps a
        spike's influence on the other side's timing window survives.
        """
        self._device = device or torch.device("cpu")
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_min = w_min
        self.w_max = w_max
        self._weight = torch.zeros(
            in_features, out_features, device=self._device
        )
        self._pre_trace = torch.zeros(in_features, device=self._device)
        self._post_trace = torch.zeros(out_features, device=self._device)

    @property
    def weight(self) -> torch.Tensor:
        """Return the current synaptic weight matrix."""
        return self._weight

    def set_weight(self, weight: torch.Tensor) -> None:
        """Overwrite the weight matrix (e.g. to seed a starting condition)."""
        self._weight = weight.to(self._device)

    def step(
        self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor
    ) -> None:
        """Advance both traces by one timestep and apply the STDP update.

        Trace decay happens first so a spike this step reads the *prior*
        step's trace when computing its update — the update a spike
        causes must depend on what came before it, not on itself.
        """
        pre_spikes = pre_spikes.to(self._device)
        post_spikes = post_spikes.to(self._device)
        with torch.no_grad():
            decay_pre = torch.exp(torch.tensor(-1.0 / self.tau_pre))
            decay_post = torch.exp(torch.tensor(-1.0 / self.tau_post))
            self._pre_trace = self._pre_trace * decay_pre
            self._post_trace = self._post_trace * decay_post

            # A post spike landing on a lingering pre trace is causal
            # (pre fired first) and potentiates; a pre spike landing on a
            # lingering post trace is anti-causal (post fired first) and
            # depresses. Each side reads the *other* trace, not its own.
            potentiation = torch.outer(self._pre_trace, post_spikes)
            depression = torch.outer(pre_spikes, self._post_trace)
            self._weight += self.a_plus * potentiation
            self._weight -= self.a_minus * depression
            self._weight.clamp_(self.w_min, self.w_max)

            self._pre_trace += pre_spikes
            self._post_trace += post_spikes

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Return the synaptic current ``spikes @ weight`` for one step."""
        return spikes.to(self._device) @ self._weight
