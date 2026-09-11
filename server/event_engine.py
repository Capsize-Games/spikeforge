"""Serve an event sample through the encoder-engine method surface."""

from typing import Any, Dict, Optional, Tuple

import torch

from server.schemas import EncodeConfig
from spikeforge.events.dense import to_frames
from spikeforge.events.event_bridge import EventSpikeBridge
from spikeforge.events.event_source import EventSampleSource
from spikeforge.topology.spec import TopologySpec


def _grid(frame: torch.Tensor) -> Any:
    """Stack the ON and OFF channels into a ``2H x W`` grid."""
    return torch.cat([frame[0], frame[1]], dim=0).tolist()


def _flat(values: torch.Tensor) -> Any:
    """Return an index tensor as a flat python list."""
    return values.detach().cpu().long().tolist()


class EventEngine:
    """Expose an event sample like :class:`~server.encoder.EncoderEngine`.

    The sample is densified once by
    :func:`~spikeforge.events.dense.to_frames` into ``[T, 2, H, W]``
    ON/OFF frames. ``spike_input`` returns the flat feature layout
    ``[T, 1, H*W]`` the image engine also returns, so the shared
    ``input_shape.to_input_shape`` consumers reshape it for spatial
    topologies; when a ``spec`` is supplied the tensor is laid out for it
    immediately through
    :class:`~spikeforge.events.event_bridge.EventSpikeBridge`.
    """

    def __init__(
        self,
        config: EncodeConfig,
        spec: Optional[TopologySpec] = None,
        source: Optional[EventSampleSource] = None,
    ) -> None:
        """Load the configured sample and densify it into ON/OFF frames."""
        self._config = config
        self._source = source or EventSampleSource(config.dataset)
        self._index = self._source.clamp(config.sample_index)
        self._sample, self._label = self._source.load(self._index)
        self._frames = to_frames(self._sample)
        self._shaped = self._shape(spec)

    def _shape(self, spec: Optional[TopologySpec]) -> Optional[torch.Tensor]:
        """Return the spec-shaped spike tensor, or None without a spec."""
        if spec is None:
            return None
        spikes, _meta = EventSpikeBridge().encode(self._sample, spec)
        return spikes

    # --- payload builders -------------------------------------------------

    def sample_frame(self) -> Any:
        """Return the whole-sample ON/OFF frame as a stacked grid."""
        return _grid(self._frames.amax(dim=0))

    def spike_frame(self, step: int) -> Any:
        """Return one ON/OFF frame as a stacked ``2H x W`` grid."""
        index = min(max(int(step), 0), self.num_steps() - 1)
        return _grid(self._frames[index])

    def raster(self, max_neurons: Optional[int] = None) -> Dict[str, Any]:
        """Return polarity-aware ``(time, neuron)`` event coordinates.

        Neuron indices ``[0, H*W)`` are ON events and ``[H*W, 2*H*W)`` are
        OFF events, so the raster reuses the shared ``RasterPayload`` shape.
        """
        height, width = self._sample.shape
        offset = height * width
        on_time, off_time = self._times()
        on_neuron, off_neuron = self._neurons()
        return {
            "time": _flat(torch.cat([on_time, off_time])),
            "neurons": _flat(torch.cat([on_neuron, off_neuron + offset])),
            "num_steps": self.num_steps(),
            "num_neurons": 2 * offset,
        }

    def _times(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the ON and OFF time coordinates of every event."""
        return (
            torch.where(self._frames[:, 0] > 0)[0],
            torch.where(self._frames[:, 1] > 0)[0],
        )

    def _neurons(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the ON and OFF in-sensor neuron coordinates."""
        return (
            torch.where(self._frames[:, 0] > 0)[1],
            torch.where(self._frames[:, 1] > 0)[1],
        )

    def num_steps(self) -> int:
        """Return the number of time bins the sample spans."""
        return int(self._frames.size(0))

    def spike_input(self) -> torch.Tensor:
        """Return the spikes in the flat ``[T, 1, H*W]`` contract."""
        if self._shaped is not None:
            return self._shaped
        return self._frames.sum(dim=1).reshape(self.num_steps(), 1, -1)

    def sample_image(self) -> None:
        """Return nothing: an event sample has no static image."""
        return

    def reconstruction(self) -> None:
        """Return nothing: events are already spikes, so no decoding runs."""
        return

    def target_label(self) -> int:
        """Return the label of the selected sample."""
        return self.sample_label()

    def sample_label(self) -> int:
        """Return the label of the selected sample."""
        return int(self._label)

    def sample_index(self) -> int:
        """Return the clamped index of the selected sample."""
        return self._index

    # --- accessors --------------------------------------------------------

    @property
    def modality(self) -> str:
        """Return the sample modality, always ``event``."""
        return "event"

    @property
    def dataset(self) -> str:
        """Return the registry key of the loaded sample source."""
        return self._source.dataset

    @property
    def source(self) -> EventSampleSource:
        """Return the sample source that served this engine."""
        return self._source
