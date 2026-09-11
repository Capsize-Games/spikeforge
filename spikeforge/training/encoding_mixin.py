"""Input encoding for the training engine (raw pixels or spike codes)."""

import torch

from spikeforge.simulator import input_shape


class EncodingMixin:
    """Turn a batch of images into spikes shaped for the active topology."""

    def _encode_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return [T,B,F] or [T,B,1,H,W] spikes per input stage."""
        return self._to_input_shape(self._flat_spikes(inputs))

    def _flat_spikes(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return flattened [T,B,784] spikes for the configured mode."""
        if self._encoder is None or self._input_mode == "raw":
            return self._repeat_pixels(inputs).to(self._device)
        if self._input_mode == "random":
            raise ValueError("random coding carries no label signal")
        return self._encoder.encode(inputs).to(self._device)

    def _to_input_shape(self, spikes: torch.Tensor) -> torch.Tensor:
        """Reshape flat spikes to the topology's input-stage layout.

        The sensor geometry comes from the topology's declared ``input_size``
        when its input stage is a convolution; a ``flatten`` input declares a
        flat feature count, and a feature/sequence topology keeps its flat
        ``[T, B, F]`` layout, so both fall back to the default geometry.
        """
        value = self._architecture.get("input_size")
        shape = input_shape.spatial_shape(self._spec, value)
        if shape is None:
            return input_shape.to_input_shape(spikes, self._spec)
        return input_shape.to_input_shape(spikes, self._spec, shape)

    def _repeat_pixels(self, inputs: torch.Tensor) -> torch.Tensor:
        """Legacy raw path: repeat normalised pixels across steps."""
        flat = inputs.view(inputs.size(0), -1)
        return flat.unsqueeze(0).repeat(self._num_steps, 1, 1)
