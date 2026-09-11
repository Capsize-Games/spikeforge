"""The timed result of a single reference NIR interpretation."""

from dataclasses import dataclass, field
from typing import Mapping

import torch


@dataclass(frozen=True)
class InterpreterResult:
    """Per-node traces and readout of an independent NIR execution.

    ``readout`` is the ``Output`` node accumulated over the run and divided
    by ``steps``, matching the simulator's logits. ``spikes`` maps every
    spiking node name to its ``[T, ...]`` spike train and ``membranes`` maps
    every integrator node name to its ``[T, ...]`` membrane trace; either is
    empty when the graph contains no node of that role.
    """

    steps: int
    readout: torch.Tensor
    spikes: Mapping[str, torch.Tensor] = field(default_factory=dict)
    membranes: Mapping[str, torch.Tensor] = field(default_factory=dict)
