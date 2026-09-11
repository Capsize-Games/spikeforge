"""Synaptic (second-order) neuron handler (``snn.Synaptic``)."""

from typing import Any, Dict, Mapping, Optional, Tuple

import snntorch as snn
import torch
import torch.nn as nn

from spikeforge.neurons import contract
from spikeforge.neurons.contract import NeuronState
from spikeforge.neurons.spike_grad import spike_grad_from


class SynapticNeuron:
    """Build an ``snn.Synaptic`` and describe it in NIR parameter terms."""

    kind: str = "synaptic"

    def build(self, params: Mapping[str, Any]) -> nn.Module:
        """Build an ``snn.Synaptic`` from ``params``."""
        return snn.Synaptic(
            alpha=contract.get_float(params, "alpha", contract.DEFAULT_ALPHA),
            beta=contract.get_float(params, "beta", contract.DEFAULT_BETA),
            threshold=contract.get_float(
                params, "threshold", contract.DEFAULT_THRESHOLD
            ),
            reset_mechanism=contract.get_reset(params),
            spike_grad=spike_grad_from(params),
        )

    def nir_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Return NIR parameters with both synaptic and membrane constants."""
        dt = contract.get_float(params, "dt", contract.DEFAULT_DT)
        alpha = contract.get_float(params, "alpha", contract.DEFAULT_ALPHA)
        base = contract.base_params(self.kind, params)
        base["tau_syn"] = contract.tau_from_decay(alpha, dt)
        return base

    def initial_state(self, device: torch.device) -> NeuronState:
        """Return the zero-length ``(syn, mem)`` state on ``device``."""
        return (torch.empty(0, device=device), torch.empty(0, device=device))

    def step(
        self,
        module: nn.Module,
        x: torch.Tensor,
        state: Optional[NeuronState],
    ) -> Tuple[torch.Tensor, NeuronState]:
        """Run one step, returning ``(spikes, (syn, mem))``."""
        if state is None:
            state = self.initial_state(x.device)
        spk, syn, mem = module(x, state[0], state[1])
        return spk, (syn, mem)
