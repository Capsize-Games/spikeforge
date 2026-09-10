"""A small fully-connected spiking neural network for image data."""

from typing import Any, Dict, List

import torch

from snn_interpreter.simulator.runner import run
from snn_interpreter.simulator.trajectory import Trajectory
from snn_interpreter.topology.presets import fc_legacy
from snn_interpreter.topology.stage_module import StageModule

#: Legacy stage names whose per-step spikes form the tracking contract.
_HIDDEN_STAGE = "_lif1"
_OUTPUT_STAGE = "_lif2"


def _frames(trace: torch.Tensor) -> List[torch.Tensor]:
    """Split a ``[T, ...]`` trace tensor into a list of per-step frames."""
    return [trace[index] for index in range(trace.size(0))]


class SpikingNet(StageModule):
    """Fully-connected LIF network emitting logits over time steps.

    This is a thin subclass of the ``StageModule`` rendering of the
    ``fc_legacy`` preset, so its submodules register under the original
    ``_fc1``/``_lif1``/``_fc2``/``_lif2`` names and its ``state_dict`` keys
    stay checkpoint-compatible. Temporal execution is delegated to the
    generic simulator.
    """

    def __init__(
        self,
        hidden: int = 128,
        beta: float = 0.5,
        num_classes: int = 10,
        input_size: int = 28 * 28,
    ) -> None:
        """Build the ``fc_legacy`` topology and record legacy attributes."""
        super().__init__(fc_legacy(hidden, beta, num_classes, input_size))
        self._hidden = hidden
        self._beta = beta
        self._num_classes = num_classes
        self._input_size = input_size

    def forward(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Legacy raw-pixel path: repeat the static frame, old behavior."""
        flat = x.view(x.size(0), -1)
        frames = flat.unsqueeze(0).repeat(num_steps, 1, 1)
        return self.forward_spikes(frames)

    def forward_spikes(
        self, spikes: torch.Tensor, track: bool = False,
        membrane: bool = False,
    ) -> Any:
        """Consume ``[T, B, F]`` spikes and return logits or a track dict.

        With ``track``/``membrane`` off the mean logits tensor is returned
        exactly as before. Recording returns the legacy dict keys
        (``logits``, ``hidden``, ``output``, ``steps``) and, when
        ``membrane`` is set, an additive ``membrane`` trace key.
        """
        record = track or membrane
        trajectory = run(self, spikes, track=record, membrane=membrane)
        if not record:
            return trajectory.logits
        return self._tracked(trajectory)

    def _tracked(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Shape a trajectory into the legacy tracking-dict contract."""
        packed: Dict[str, Any] = {
            "logits": trajectory.logits,
            "hidden": _frames(trajectory.spikes[_HIDDEN_STAGE]),
            "output": _frames(trajectory.spikes[_OUTPUT_STAGE]),
            "steps": trajectory.steps,
        }
        if trajectory.membranes:
            packed["membrane"] = {
                name: _frames(trace)
                for name, trace in trajectory.membranes.items()
            }
        return packed

    @property
    def hidden(self) -> int:
        """Return the hidden-layer width."""
        return self._hidden

    @property
    def beta(self) -> float:
        """Return the LIF membrane decay rate."""
        return self._beta

    @property
    def num_classes(self) -> int:
        """Return the number of output classes."""
        return self._num_classes

    @property
    def input_size(self) -> int:
        """Return the flattened input features per time step."""
        return self._input_size
