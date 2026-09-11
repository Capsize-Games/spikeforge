"""Lapicque neuron handler (``snn.Lapicque``)."""

from typing import Any, Dict, Mapping, Optional, Tuple

import snntorch as snn
import torch
import torch.nn as nn

from snn_interpreter.neurons import contract
from snn_interpreter.neurons.contract import NeuronState
from snn_interpreter.neurons.spike_grad import spike_grad_from


class LapicqueNeuron:
    """Build an ``snn.Lapicque`` and describe it in NIR parameter terms."""

    kind: str = "lapicque"

    def build(self, params: Mapping[str, Any]) -> nn.Module:
        """Build an ``snn.Lapicque`` from ``params``."""
        return snn.Lapicque(
            beta=contract.get_float(params, "beta", contract.DEFAULT_BETA),
            threshold=contract.get_float(
                params, "threshold", contract.DEFAULT_THRESHOLD
            ),
            reset_mechanism=contract.get_reset(params),
            spike_grad=spike_grad_from(params),
        )

    def nir_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Return the canonical NIR parameters for a Lapicque neuron."""
        return contract.base_params(self.kind, params)

    def initial_state(self, device: torch.device) -> NeuronState:
        """Return the zero-length membrane state on ``device``."""
        return (torch.empty(0, device=device),)

    def step(
        self,
        module: nn.Module,
        x: torch.Tensor,
        state: Optional[NeuronState],
    ) -> Tuple[torch.Tensor, NeuronState]:
        """Run one step, returning ``(spikes, (mem,))``."""
        if state is None:
            state = self.initial_state(x.device)
        spk, mem = module(x, state[0])
        return spk, (mem,)
