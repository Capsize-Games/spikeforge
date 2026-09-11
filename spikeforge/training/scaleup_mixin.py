"""Opt-in training scale-ups: AMP, checkpointing, BPTT, and multi-GPU.

Every option is additive and defaults off, so the forward path is a thin
wrapper that reproduces the previous behaviour exactly when nothing is
enabled. ``_run_trajectory`` (defined on the engine, which owns the ``run``
import) performs the graded forward; this mixin only decides whether that
returns directly or goes through a DataParallel fan-out.
"""

from typing import Any, Dict, Optional

import torch
from torch.nn.functional import cross_entropy

from spikeforge.observability import metrics
from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.training.amp_controller import AmpController
from spikeforge.training.multi_device import MultiDeviceManager


def _window(bptt_steps: Optional[int]) -> Optional[int]:
    """Normalise a BPTT window, treating non-positive values as disabled."""
    if bptt_steps is None:
        return None
    steps = int(bptt_steps)
    return steps if steps > 0 else None


def scaleup_options(
    amp: bool,
    grad_checkpoint: bool,
    bptt_steps: Optional[int],
    multi_gpu: bool,
) -> Dict[str, Any]:
    """Bundle the additive scale-up flags for the engine."""
    return {
        "amp": bool(amp),
        "grad_checkpoint": bool(grad_checkpoint),
        "bptt_steps": bptt_steps,
        "multi_gpu": bool(multi_gpu),
    }


class ScaleUpMixin:
    """Configure and apply the opt-in training scale-up options."""

    _device: torch.device
    _mode: ExecutionMode
    _optimizer: torch.optim.Adam

    _amp: AmpController
    _grad_checkpoint: bool
    _bptt_steps: Optional[int]
    _multi: MultiDeviceManager
    _runner: Optional[Any]

    def _configure_scaleups(
        self,
        amp: bool,
        grad_checkpoint: bool,
        bptt_steps: Optional[int],
        multi_gpu: bool,
    ) -> None:
        """Resolve the AMP policy, grad options, and multi-GPU decision."""
        self._amp = AmpController(amp, self._device)
        self._grad_checkpoint = bool(grad_checkpoint)
        self._bptt_steps = _window(bptt_steps)
        self._multi = MultiDeviceManager(multi_gpu, self._device)
        self._runner = self._multi.runner(
            self._net,
            mode=self._mode,
            grad_checkpoint=self._grad_checkpoint,
            bptt_steps=self._bptt_steps,
        )

    # --- forward ---------------------------------------------------------

    def _logits(self, spikes: torch.Tensor) -> torch.Tensor:
        """Return readout logits, fanning out only when multi-GPU is active."""
        if self._runner is None:
            return self._run_trajectory(spikes).logits
        return self._runner(spikes.transpose(0, 1))

    # --- training --------------------------------------------------------

    def _train_batch(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Run one optimisation step and return loss/accuracy."""
        targets = targets.to(self._device)
        with self._amp.autocast():
            with metrics.timer("train.encode_seconds"):
                spikes = self._encode_batch(inputs)
            with metrics.timer("train.forward_seconds"):
                logits = self._logits(spikes)
            loss = cross_entropy(logits, targets)
        with metrics.timer("train.backward_seconds"):
            self._amp.backward(loss, self._optimizer)
        metrics.counter("train.steps")
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        return {"loss": float(loss.item()), "train_accuracy": float(accuracy)}

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return predicted digits for a batch of images."""
        with torch.no_grad():
            logits = self._logits(self._encode_batch(inputs))
        return logits.argmax(dim=1)

    # --- accessors -------------------------------------------------------

    @property
    def amp(self) -> bool:
        """Return True when autocast AMP actually runs."""
        return self._amp.enabled

    @property
    def amp_dtype(self) -> Optional[str]:
        """Return the active AMP dtype name, or None when AMP is off."""
        return self._amp.dtype

    @property
    def grad_checkpoint(self) -> bool:
        """Return True when per-step activation checkpointing is active."""
        return self._grad_checkpoint

    @property
    def bptt_steps(self) -> Optional[int]:
        """Return the BPTT window, or None for full backprop-through-time."""
        return self._bptt_steps

    @property
    def multi_gpu(self) -> bool:
        """Return True when DataParallel fan-out is active."""
        return self._multi.active

    @property
    def multi_gpu_status(self) -> str:
        """Return the honest multi-GPU decision string."""
        return self._multi.status

    def scale_up_status(self) -> Dict[str, Any]:
        """Return the JSON-able scale-up status block for payloads."""
        return {
            "amp": self.amp,
            "amp_dtype": self.amp_dtype,
            "grad_checkpoint": self._grad_checkpoint,
            "bptt_steps": self._bptt_steps,
            "multi_gpu": self.multi_gpu,
            "multi_gpu_status": self.multi_gpu_status,
        }
