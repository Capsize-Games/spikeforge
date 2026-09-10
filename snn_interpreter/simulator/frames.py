"""Normalisation of raw simulator frames to an input stage's shape."""

from typing import Tuple

import torch

#: Kinds whose input stage consumes multi-dimensional (spatial) frames.
SPATIAL_KINDS: Tuple[str, ...] = (
    "conv2d",
    "avgpool2d",
    "sumpool2d",
    "flatten",
)


def normalise_frame(frame: torch.Tensor, kind: str) -> torch.Tensor:
    """Reshape ``frame`` to the shape an input stage of ``kind`` expects.

    Feature stages (``linear`` and neurons) flatten every trailing axis into
    one, while spatial stages keep their channel/height/width layout and let
    the stage itself handle flattening or pooling.
    """
    if kind in SPATIAL_KINDS:
        return frame
    return frame.reshape(frame.size(0), -1)
