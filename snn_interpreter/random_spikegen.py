"""Random spike generation from scratch (tutorial 3)."""

import torch
from snntorch import spikegen


class RandomSpikeGenerator:
    """Generate random spike trains without any source data."""

    _num_steps = 100
    _size = (28, 28)
    _scale = 0.5
    _dtype = torch.float
    _spike_rand = None

    def __init__(self, num_steps=100, size=(28, 28), scale=0.5):
        self._num_steps = num_steps
        self._size = size
        self._scale = scale
        self._generate()

    def _generate(self):
        """Create a random spike train from a uniform probability grid."""
        shape = (self._num_steps, *self._size)
        spike_prob = torch.rand(shape, dtype=self._dtype) * self._scale
        self._spike_rand = spikegen.rate_conv(spike_prob)

    @property
    def spike_rand(self):
        return self._spike_rand

    @property
    def num_steps(self):
        return self._num_steps

    @property
    def size(self):
        return self._size

    @property
    def scale(self):
        return self._scale
