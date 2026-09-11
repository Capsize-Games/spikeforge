"""Per-step hidden-layer activation frames for the client animation.

The simulator already captures per-step hidden spikes when a trajectory is
tracked; this module turns that same trace into a stream of 2D frames the
client can play back. Only formatting lives here: each frame is a single-row
``[1, N]`` grid over the hidden stage's neurons, capped so an oversized layer
cannot flood the socket.
"""

from typing import Any, List

import torch

from snn_interpreter.network.inference import hidden_stage
from snn_interpreter.runtime.execution_mode import ExecutionMode
from snn_interpreter.simulator.module_spec import spec_of
from snn_interpreter.simulator.runner import run

#: Largest hidden-layer width emitted in one animation frame.
HIDDEN_CAP = 256
#: One animation frame: a single row of per-neuron activations.
Frame = List[List[float]]


def _row(step: torch.Tensor, width: int) -> Frame:
    """Return one step's hidden activations as a single-row grid."""
    values = step[0].reshape(-1)[:width].detach().float()
    return [[float(x) for x in values]]


def hidden_frame_series(net: Any, spikes: torch.Tensor) -> List[Frame]:
    """Return one ``[1, N]`` activation frame per step for ``net``'s hidden."""
    trajectory = run(net, spikes, track=True, mode=ExecutionMode.PRODUCTION)
    trace = trajectory.spikes[hidden_stage(spec_of(net))]
    flat = trace.reshape(trace.size(0), trace.size(1), -1)
    width = min(int(flat.size(-1)), HIDDEN_CAP)
    return [_row(flat[index], width) for index in range(flat.size(0))]
