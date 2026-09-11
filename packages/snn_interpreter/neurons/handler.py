"""Structural type implemented by every spiking neuron handler."""

from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

import torch
import torch.nn as nn

from snn_interpreter.neurons.contract import NeuronState


class NeuronHandler(Protocol):
    """Build, step, and describe one snnTorch neuron model."""

    kind: str

    def build(self, params: Mapping[str, Any]) -> nn.Module:
        """Return the snnTorch module for ``params``."""
        ...

    def nir_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Return the canonical NIR parameter contract for ``params``."""
        ...

    def initial_state(self, device: torch.device) -> NeuronState:
        """Return the zero-length hidden state on ``device``."""
        ...

    def step(
        self,
        module: nn.Module,
        x: torch.Tensor,
        state: Optional[NeuronState],
    ) -> Tuple[torch.Tensor, NeuronState]:
        """Advance ``module`` one step, returning spikes and new state."""
        ...
