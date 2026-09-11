"""Bridge sparse event samples to the simulator's spike tensor contract.

For image datasets the existing ``SpikeEncoder`` remains the encoding path;
this bridge is the event-modality path. It never re-implements coding math:
the dense accumulation lives in :mod:`snn_interpreter.events.dense`, the
layout decision reuses :func:`simulator.input_shape.is_spatial`, and
sparsity is reported through the shared introspection helper. Feature-input
topologies receive ``[T, B, F]`` and spatial input stages ``[T, B, C, H, W]``,
exactly the two layouts :func:`simulator.runner.run` consumes.
"""

from typing import Any, Dict, Tuple

import torch

from snn_interpreter.events.dense import to_frames, to_voxel
from snn_interpreter.events.event_sample import EventSample
from snn_interpreter.introspection.sparsity import sparsity
from snn_interpreter.simulator.input_shape import (
    SAMPLE_CHANNELS,
    is_spatial,
    to_input_shape,
)
from snn_interpreter.topology.spec import TopologySpec

#: Dense accumulation modes exposed by the bridge.
MODES: Tuple[str, ...] = ("binary", "count")


class EventSpikeBridge:
    """Convert an event sample into a simulator-ready spike tensor.

    ``mode`` selects the dense accumulation: ``binary`` uses
    :func:`~snn_interpreter.events.dense.to_frames` and ``count`` the raw
    spike-count voxel from :func:`~snn_interpreter.events.dense.to_voxel`.
    The bridge always emits batch size ``1``; callers stack samples
    themselves if they need a larger batch.
    """

    def __init__(self, mode: str = "binary") -> None:
        """Store the dense accumulation mode (``binary`` or ``count``)."""
        self._mode = mode if mode in MODES else "binary"

    def encode(
        self, sample: EventSample, spec: TopologySpec
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return ``(spikes, metadata)`` for ``sample`` on ``spec``.

        ``spikes`` is the exact layout ``spec``'s input stage expects and
        ``metadata`` is plain, JSON-able conversion metadata.
        """
        frames = self._frames(sample)
        spikes = self._layout(frames, spec)
        return spikes, self._metadata(sample, spec, spikes)

    def _frames(self, sample: EventSample) -> torch.Tensor:
        """Return the ``[T, 2, H, W]`` float accumulation for ``sample``."""
        if self._mode == "count":
            return to_voxel(sample).float()
        return to_frames(sample)

    def _layout(
        self, frames: torch.Tensor, spec: TopologySpec
    ) -> torch.Tensor:
        """Return ``frames`` in the input layout ``spec`` consumes."""
        stacked = frames.unsqueeze(1)
        if self._channels(spec) != 2:
            stacked = stacked.sum(dim=2, keepdim=True)
        if is_spatial(spec):
            return self._spatial(stacked, spec)
        return stacked.reshape(stacked.size(0), stacked.size(1), -1)

    def _spatial(
        self, frames: torch.Tensor, spec: TopologySpec
    ) -> torch.Tensor:
        """Return ``[T, B, C, H, W]`` frames for a spatial input stage.

        A single-channel, square frame goes through the shared
        :func:`to_input_shape` reshape; a polar multi-channel frame (or a
        non-square one) is already laid out correctly and passes through.
        """
        square = frames.size(3) == frames.size(4)
        if frames.size(2) == SAMPLE_CHANNELS and square:
            flat = frames.reshape(frames.size(0), frames.size(1), -1)
            return to_input_shape(flat, spec, size=frames.size(3))
        return frames

    def _channels(self, spec: TopologySpec) -> int:
        """Return the input stage's channel count (one when unspecified)."""
        params = spec.stage(spec.input).params
        return int(params.get("in_channels", SAMPLE_CHANNELS))

    def _metadata(
        self, sample: EventSample, spec: TopologySpec, spikes: torch.Tensor
    ) -> Dict[str, Any]:
        """Return plain, JSON-able metadata describing the conversion."""
        height, width = sample.shape
        return {
            "modality": "event",
            "mode": self._mode,
            "shape": [height, width],
            "bins": sample.num_steps,
            "num_events": sample.num_events,
            "channels": self._channels(spec),
            "spatial": is_spatial(spec),
            "topology_input": spec.input,
            "output_shape": list(spikes.shape),
            "sparsity": sparsity(spikes),
        }
