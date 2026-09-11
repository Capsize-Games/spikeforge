"""Normalisation of raw simulator frames to an input stage's shape."""

from typing import Tuple

import torch

#: Kinds whose input stage consumes multi-dimensional (spatial) frames.
SPATIAL_KINDS: Tuple[str, ...] = (
    "conv2d",
    "maxpool2d",
    "avgpool2d",
    "sumpool2d",
    "flatten",
)
#: Kinds whose input stage consumes a ``[B, L, D]`` sequence frame.
SEQUENCE_KINDS: Tuple[str, ...] = (
    "embedding",
    "conv1d",
    "positional_encoding",
    "attention",
    "multihead_attention",
)


def normalise_frame(frame: torch.Tensor, kind: str) -> torch.Tensor:
    """Reshape ``frame`` to the shape an input stage of ``kind`` expects.

    Spatial stages keep their channel/height/width layout and sequence stages
    keep their ``[B, L, D]`` token layout. Any other multi-dimensional frame
    (for example a sequence flowing into a ``linear`` token stage) passes
    through unchanged; only a flat feature frame is collapsed, which is the
    identity for the historical ``[B, F]`` inputs.
    """
    if kind in SPATIAL_KINDS or kind in SEQUENCE_KINDS:
        return frame
    if frame.dim() >= 3:
        return frame
    return frame.reshape(frame.size(0), -1)
