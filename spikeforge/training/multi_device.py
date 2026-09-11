"""Opt-in multi-GPU execution with an honest single-device fallback.

The manager activates only on a machine with more than one visible CUDA device
when multi-GPU is explicitly requested. Otherwise it records *why* the run
stayed on one device so the caller can surface an honest status rather than
failing. Defaults are single-device and unchanged.
"""

from typing import Any, Optional

import torch

from spikeforge.simulator.parallel_runner import ParallelRunner

#: Status prefix marking an active DataParallel fan-out.
_ACTIVE = "active"


def cuda_device_count() -> int:
    """Return the number of visible CUDA devices (0 without CUDA)."""
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


class MultiDeviceManager:
    """Decide whether to fan a run across CUDA devices, and report it."""

    def __init__(self, enabled: bool, device: torch.device) -> None:
        """Record whether multi-GPU is active and the reason either way."""
        self._device = device
        self._count = cuda_device_count()
        self._enabled = bool(enabled)
        self._status = self._resolve()

    def _resolve(self) -> str:
        """Return a human-readable status string for the decision."""
        if not self._enabled:
            return "disabled"
        if self._device.type != "cuda":
            return f"unavailable: device is {self._device.type}"
        if self._count <= 1:
            return f"unavailable: {self._count} cuda device(s) visible"
        return f"{_ACTIVE}: {self._count} cuda devices"

    @property
    def active(self) -> bool:
        """Return True when the run should use DataParallel."""
        return self._status.startswith(_ACTIVE)

    @property
    def status(self) -> str:
        """Return the honest multi-GPU status string."""
        return self._status

    def runner(self, module: Any, **kwargs: Any) -> Optional[Any]:
        """Wrap ``module`` for DataParallel, or return None when inactive."""
        if not self.active:
            return None
        runner = ParallelRunner(module, **kwargs)
        return torch.nn.DataParallel(runner).to(self._device)
