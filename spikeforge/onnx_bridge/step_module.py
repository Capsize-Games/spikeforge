"""A one-step ``nn.Module`` adapter used to export a topology to ONNX.

A :class:`~spikeforge.topology.stage_module.StageModule` exposes
``step(x, state) -> (outputs, state)`` rather than a tensor ``forward``, and
ONNX can only record tensor-to-tensor graphs. This adapter runs exactly one
step from a fresh state and returns the readout stage's tensor, which is the
"one step, external loop" temporal contract the bridge documents.
"""

import torch
import torch.nn as nn

from spikeforge.topology.stage_module import StageModule


class StepModule(nn.Module):
    """Expose a built module's single-step readout as a tensor ``forward``."""

    def __init__(self, module: StageModule, output: str) -> None:
        """Wrap ``module`` and select the ``output`` stage's tensor."""
        super().__init__()
        self.module = module
        self.output = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the readout stage's tensor after one step from rest."""
        outputs, _ = self.module.step(x, None)
        return outputs[self.output]
