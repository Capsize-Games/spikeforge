"""The shared stand-in for the configured encoder engine.

Five server-action test files carried their own ``FakeSample``; this is
their union. It serves the ``[T,1,784]`` spike volume every handler
reads, plus the raw image and encoder the introspection handlers need.
"""

import torch

from spikeforge.encoding.spike_encoder import SpikeEncoder


class FakeSample:
    """A minimal stand-in for the configured encoder engine."""

    def __init__(self, steps: int = 4) -> None:
        """Hold a fixed spike volume, image, and encoder."""
        self._spikes = torch.rand(steps, 1, 784)
        self._image = torch.rand(1, 28, 28)
        self._encoder = SpikeEncoder(coding="rate", num_steps=steps)

    def spike_input(self) -> torch.Tensor:
        """Return the ``[T,1,784]`` spike volume."""
        return self._spikes

    def sample_tensor(self) -> torch.Tensor:
        """Return the raw ``[1,28,28]`` image."""
        return self._image

    @property
    def encoder(self) -> SpikeEncoder:
        """Return the encoder that produced the spikes."""
        return self._encoder
