"""A small fully-connected spiking neural network for image data."""

from typing import Any, List, Tuple

import snntorch as snn
import torch
import torch.nn as nn


class SpikingNet(nn.Module):
    """Fully-connected LIF network emitting logits over time steps."""

    def __init__(
        self,
        hidden: int = 128,
        beta: float = 0.5,
        num_classes: int = 10,
        input_size: int = 28 * 28,
    ) -> None:
        """Build the two linear layers and their LIF neurons."""
        super().__init__()
        self._hidden = hidden
        self._beta = beta
        self._num_classes = num_classes
        self._input_size = input_size
        self._fc1 = nn.Linear(input_size, hidden)
        self._lif1 = snn.Leaky(beta=beta)
        self._fc2 = nn.Linear(hidden, num_classes)
        self._lif2 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Legacy raw-pixel path: repeat the static frame, old behavior."""
        flat = x.view(x.size(0), -1)
        frames = flat.unsqueeze(0).repeat(num_steps, 1, 1)
        return self.forward_spikes(frames)

    def forward_spikes(self, spikes: torch.Tensor, track: bool = False) -> Any:
        """Consume [T,B,F] spikes, re-injecting frame t each step."""
        frames = spikes.reshape(spikes.size(0), spikes.size(1), -1)
        device = frames.device
        mem1 = self._lif1.init_leaky().to(device)
        mem2 = self._lif2.init_leaky().to(device)
        out_sum = torch.zeros(spikes.size(1), self._num_classes, device=device)
        hidden: List[torch.Tensor] = []
        output: List[torch.Tensor] = []
        for t in range(frames.size(0)):
            spk1, mem1, spk2, mem2 = self._step(frames[t], mem1, mem2)
            out_sum = out_sum + spk2
            if track:
                hidden.append(spk1)
                output.append(spk2)
        return self._pack(out_sum, frames.size(0), hidden, output, track)

    def _step(
        self, x_t: torch.Tensor, mem1: torch.Tensor, mem2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance both LIF layers one time step."""
        spk1, mem1 = self._lif1(self._fc1(x_t), mem1)
        spk2, mem2 = self._lif2(self._fc2(spk1), mem2)
        return spk1, mem1, spk2, mem2

    def _pack(
        self,
        out_sum: torch.Tensor,
        steps: int,
        hidden: List[torch.Tensor],
        output: List[torch.Tensor],
        track: bool,
    ) -> Any:
        """Return logits tensor, or a tracking dict when requested."""
        logits = out_sum / steps
        if not track:
            return logits
        return {
            "logits": logits,
            "hidden": hidden,
            "output": output,
            "steps": steps,
        }

    @property
    def hidden(self) -> int:
        """Return the hidden-layer width."""
        return self._hidden

    @property
    def beta(self) -> float:
        """Return the LIF membrane decay rate."""
        return self._beta

    @property
    def num_classes(self) -> int:
        """Return the number of output classes."""
        return self._num_classes

    @property
    def input_size(self) -> int:
        """Return the flattened input features per time step."""
        return self._input_size
