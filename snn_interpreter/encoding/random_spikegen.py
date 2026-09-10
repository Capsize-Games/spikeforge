"""Random spike generation from scratch (tutorial 3)."""

from typing import Optional, Tuple

import torch
from snntorch import spikegen


class RandomSpikeGenerator:
    """Generate random spike trains without any source data."""

    _num_steps: int = 100
    _size: Tuple[int, int] = (28, 28)
    _scale: float = 0.5
    _dtype: torch.dtype = torch.float
    _spike_rand: Optional[torch.Tensor] = None

    def __init__(
        self,
        num_steps: int = 100,
        size: Tuple[int, int] = (28, 28),
        scale: float = 0.5,
    ) -> None:
        """Store the shape/scale options and generate the spike train."""
        self._num_steps = num_steps
        self._size = size
        self._scale = scale
        self._generate()

    def _generate(self) -> None:
        """Create a random spike train from a uniform probability grid."""
        shape = (self._num_steps, *self._size)
        spike_prob = torch.rand(shape, dtype=self._dtype) * self._scale
        self._spike_rand = spikegen.rate_conv(spike_prob)

    @property
    def spike_rand(self) -> Optional[torch.Tensor]:
        """Return the generated random spike train."""
        return self._spike_rand

    @property
    def num_steps(self) -> int:
        """Return the number of generated time steps."""
        return self._num_steps

    @property
    def size(self) -> Tuple[int, int]:
        """Return the (H, W) grid size of the spikes."""
        return self._size

    @property
    def scale(self) -> float:
        """Return the probability scale of the noise grid."""
        return self._scale
