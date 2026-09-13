"""Slow recurrent spiking state for downstream gain modulation."""

from typing import NamedTuple, Optional

import snntorch as snn
import torch
from torch import nn


class AffectiveOutput(NamedTuple):
    """One regulator step's spikes and continuous broadcast signal."""

    spikes: torch.Tensor
    broadcast: torch.Tensor


class AffectiveRegulator(nn.Module):
    """Integrate inputs into a slowly decaying SNN-native affect state.

    The linear projection turns signals such as novelty or confidence into
    currents. A recurrent LIF membrane integrates those currents and is
    exposed as the continuous ``broadcast`` signal; spikes are provided for
    consumers that want an event-native modulation path.
    """

    def __init__(
        self,
        input_size: int,
        state_size: int = 2,
        beta: float = 0.98,
        threshold: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """Create a regulator with a persistent state per batch item."""
        super().__init__()
        if input_size < 1 or state_size < 1:
            raise ValueError("input_size and state_size must be positive")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be between 0 and 1")
        self.projection = nn.Linear(
            input_size, state_size, bias=False, device=device,
        )
        self.neuron = snn.Leaky(
            beta=beta,
            threshold=threshold,
            reset_mechanism="subtract",
        ).to(device)
        self._mem: Optional[torch.Tensor] = None

    @property
    def state_size(self) -> int:
        """Return the number of broadcast channels."""
        return self.projection.out_features

    def reset_state(self, batch_size: int = 1) -> None:
        """Clear the carried state before a new sequence."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        device = self.projection.weight.device
        self._mem = torch.zeros(batch_size, self.state_size, device=device)

    def step(self, inputs: torch.Tensor) -> AffectiveOutput:
        """Advance the regulator for inputs shaped ``[B, input_size]``."""
        if inputs.ndim != 2 or inputs.size(1) != self.projection.in_features:
            raise ValueError(
                "inputs must have shape [B, input_size]"
            )
        batch_size = inputs.size(0)
        if self._mem is None or self._mem.size(0) != batch_size:
            self.reset_state(batch_size)
        current = self.projection(inputs)
        spikes, self._mem = self.neuron(current, self._mem)
        return AffectiveOutput(spikes=spikes, broadcast=self._mem)

    def forward(self, inputs: torch.Tensor) -> AffectiveOutput:
        """Alias :meth:`step` for ordinary ``nn.Module`` use."""
        return self.step(inputs)
