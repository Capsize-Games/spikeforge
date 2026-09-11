"""DataParallel-friendly wrapper that runs the temporal loop per replica.

``torch.nn.DataParallel`` scatters its input along dimension 0 and gathers the
output back along dimension 0, so the batch must lead here even though the
simulator consumes ``[T, B, ...]`` trains. The forward transposes a
``[B, T, ...]`` batch into a ``[T, B, ...]`` train, runs the single shared loop
on that replica's slice, and returns ``[B, C]`` logits for the gather. Each
replica drives its own device-local neuron state, so the loop stays unchanged.
"""

from typing import Optional

import torch

from snn_interpreter.runtime.execution_mode import ExecutionMode
from snn_interpreter.simulator.runner import run
from snn_interpreter.topology.stage_module import StageModule


class ParallelRunner(torch.nn.Module):
    """Run the shared temporal loop over a batch-first spike train."""

    def __init__(
        self,
        module: StageModule,
        mode: ExecutionMode = ExecutionMode.PRODUCTION,
        grad_checkpoint: bool = False,
        bptt_steps: Optional[int] = None,
    ) -> None:
        """Register ``module`` so DataParallel can replicate its weights."""
        super().__init__()
        self.inner = module
        self._mode = mode
        self._grad_checkpoint = bool(grad_checkpoint)
        self._bptt_steps = bptt_steps

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Return ``[B, C]`` logits for a ``[B, T, ...]`` spike batch."""
        return run(
            self.inner,
            spikes.transpose(0, 1),
            mode=self._mode,
            grad_checkpoint=self._grad_checkpoint,
            bptt_steps=self._bptt_steps,
        ).logits
