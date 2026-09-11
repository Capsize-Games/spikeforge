"""Reshape flat spike trains to a topology's input-stage layout."""

from typing import Any, Optional, Tuple

import torch

from spikeforge.data.image_size import SizeLike, as_size
from spikeforge.simulator.frames import SEQUENCE_KINDS, SPATIAL_KINDS
from spikeforge.topology.spec import TopologySpec

#: Default sample geometry: the registry's 1x28x28 grayscale frame.
SAMPLE_SIZE: Tuple[int, int] = (28, 28)
SAMPLE_CHANNELS = 1
#: Input-stage kinds whose declared ``input_size`` is an ``(H, W)`` geometry.
SHAPED_KINDS: Tuple[str, ...] = (
    "conv2d",
    "maxpool2d",
    "avgpool2d",
    "sumpool2d",
)


def is_spatial(spec: TopologySpec) -> bool:
    """Return True when ``spec``'s input stage consumes spatial frames."""
    return spec.stage(spec.input).kind in SPATIAL_KINDS


def is_sequence(spec: TopologySpec) -> bool:
    """Return True when any stage consumes a ``[B, L, D]`` sequence frame."""
    return any(stage.kind in SEQUENCE_KINDS for stage in spec.stages)


def spatial_shape(
    spec: TopologySpec, value: Any
) -> Optional[Tuple[int, int]]:
    """Return ``spec``'s declared ``(H, W)`` geometry, or None when flat.

    A convolution-like input stage declares its sensor as a side or an
    ``(H, W)`` pair; any other spatial input (notably ``flatten``) declares a
    flat feature count, so it is not a geometry and returns ``None``.
    """
    if spec.stage(spec.input).kind not in SHAPED_KINDS:
        return None
    return as_size(SAMPLE_SIZE if value is None else value)


def to_input_shape(
    spikes: torch.Tensor, spec: TopologySpec, size: SizeLike = SAMPLE_SIZE
) -> torch.Tensor:
    """Return ``spikes`` shaped for ``spec``'s input stage.

    Spatial input stages receive ``[T, B, C, H, W]`` because the encoder
    always hands back a flattened ``[T, B, H*W]`` code; ``size`` is the sensor
    geometry, an ``int`` square side or an explicit ``(H, W)`` pair. Sequence
    topologies keep their caller-supplied ``[T, B, L, D]`` layout and feature
    input topologies keep the flat ``[T, B, F]`` layout, both unchanged.
    """
    if not is_spatial(spec):
        return spikes
    height, width = as_size(size)
    return spikes.reshape(
        spikes.size(0), spikes.size(1), SAMPLE_CHANNELS, height, width
    )
