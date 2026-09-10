"""A small fully-connected spiking neural network for image data."""

import snntorch as snn
import torch
import torch.nn as nn


class SpikingNet(nn.Module):
    """Fully-connected LIF network emitting logits over time steps."""

    def __init__(self, hidden=128, beta=0.5, num_classes=10,
                 input_size=28 * 28):
        super().__init__()
        self._hidden = hidden
        self._beta = beta
        self._num_classes = num_classes
        self._input_size = input_size
        self._fc1 = nn.Linear(input_size, hidden)
        self._lif1 = snn.Leaky(beta=beta)
        self._fc2 = nn.Linear(hidden, num_classes)
        self._lif2 = snn.Leaky(beta=beta)

    def forward(self, x, num_steps):
        """Run the network for num_steps, returning summed output spikes."""
        flat = x.view(x.size(0), -1)
        mem1 = self._lif1.init_leaky()
        mem2 = self._lif2.init_leaky()
        out_sum = torch.zeros(x.size(0), self._num_classes)
        for _ in range(num_steps):
            spk1, mem1 = self._lif1(self._fc1(flat), mem1)
            spk2, mem2 = self._lif2(self._fc2(spk1), mem2)
            out_sum = out_sum + spk2
        return out_sum / num_steps

    @property
    def hidden(self):
        return self._hidden

    @property
    def beta(self):
        return self._beta

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def input_size(self):
        return self._input_size
