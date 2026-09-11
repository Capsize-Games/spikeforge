"""Render raw samples and encoded spike volumes from a client config."""

from typing import Any, Dict, Optional

import torch

from server.schemas import EncodeConfig
from spikeforge.data.sample_source import SampleSource
from spikeforge.encoding.spike_encoder import SpikeEncoder


def to_list(tensor: torch.Tensor) -> Any:
    """Convert a tensor to a nested python list of floats."""
    return tensor.detach().cpu().float().tolist()


class EncoderEngine:
    """Build a dataset-backed sample and its encoded spike volume."""

    def __init__(self, config: EncodeConfig) -> None:
        """Encode the selected sample under the client's config."""
        self._config = config
        self._source = SampleSource(config.dataset, size=config.input_size)
        self._index = self._source.clamp(config.sample_index)
        self._encoder = SpikeEncoder.from_encode_config(config)
        self._image = self._source.image(self._index)
        self._spikes = self._encoder.encode_image(self._image)

    # --- payload builders -------------------------------------------------

    def sample_image(self) -> Any:
        """Return the raw input image grid as a nested list."""
        return to_list(self._image[0])

    def sample_tensor(self) -> torch.Tensor:
        """Return the raw input image as a ``[C, H, W]`` tensor."""
        return self._image

    def reconstruction(self) -> Optional[Dict[str, Any]]:
        """Return averaged spike reconstructions for the rate coding."""
        if self._config.coding != "rate":
            return None
        height, width = self._source.size
        gain1 = self._spikes.mean(dim=0)[0].reshape(height, width)
        low = gain1 * self._config.gain
        return {
            "gain1": to_list(gain1),
            "low": to_list(low),
            "size": [height, width],
        }

    def spike_frame(self, step: int) -> Any:
        """Return one 2-D spike frame of the selected sample."""
        height, width = self._source.size
        index = min(max(int(step), 0), self._spikes.size(0) - 1)
        return to_list(self._spikes[index, 0].reshape(height, width))

    def spike_tensor(self) -> torch.Tensor:
        """Return the full [T,1,784] spike volume."""
        return self._spikes

    def raster(self, max_neurons: int = 784) -> Dict[str, Any]:
        """Return (time, neuron) spike coordinates for the sample."""
        matrix = self._spikes[:, 0, :]
        time_idx, neuron_idx = torch.where(matrix > 0)
        return {
            "time": to_list(time_idx),
            "neurons": to_list(neuron_idx),
            "num_steps": int(matrix.size(0)),
            "num_neurons": int(matrix.size(1)),
        }

    def num_steps(self) -> int:
        """Return the number of encoded time steps."""
        return int(self._spikes.size(0))

    def target_label(self) -> int:
        """Return the label of the selected sample."""
        return self.sample_label()

    def sample_label(self) -> int:
        """Return the label of the selected sample."""
        return self._source.label(self._index)

    def sample_index(self) -> int:
        """Return the clamped index of the selected sample."""
        return self._index

    def spike_input(self) -> torch.Tensor:
        """Return the [T,1,784] spikes for training/inference."""
        return self._spikes

    @property
    def encoder(self) -> SpikeEncoder:
        """Return the encoder that produced the sample's spikes."""
        return self._encoder

    @property
    def dataset(self) -> str:
        """Return the registry key of the loaded sample source."""
        return self._source.dataset

    @property
    def modality(self) -> str:
        """Return the sample modality, always ``image``."""
        return "image"
