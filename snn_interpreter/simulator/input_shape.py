"""Reshape flat spike trains to a topology's input-stage layout."""

import torch

from snn_interpreter.simulator.frames import SPATIAL_KINDS
from snn_interpreter.topology.spec import TopologySpec

#: Registry datasets are all 1x28x28 grayscale, so flat codes reshape thus.
SAMPLE_SIDE = 28
SAMPLE_CHANNELS = 1


def is_spatial(spec: TopologySpec) -> bool:
    """Return True when ``spec``'s input stage consumes spatial frames."""
    return spec.stage(spec.input).kind in SPATIAL_KINDS


def to_input_shape(
    spikes: torch.Tensor, spec: TopologySpec, side: int = SAMPLE_SIDE
) -> torch.Tensor:
    """Return ``spikes`` shaped for ``spec``'s input stage.

    Feature-input topologies keep the flat ``[T, B, F]`` layout. Spatial
    input stages receive ``[T, B, C, H, W]`` because the encoder always hands
    back a flattened ``[T, B, 784]`` code.
    """
    if not is_spatial(spec):
        return spikes
    return spikes.reshape(
        spikes.size(0), spikes.size(1), SAMPLE_CHANNELS, side, side
    )
