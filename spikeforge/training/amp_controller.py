"""Opt-in automatic mixed precision (AMP) for the training engine.

Disabled by default, in which case the controller is a pure no-op and numerics
are unchanged. When enabled it picks ``float16`` on CUDA and ``bfloat16`` on
CPU, probing a tiny autocast op first and falling back to fp32 (``dtype`` is
``None``) when the dtype is unsupported on the device.

A ``torch.amp.GradScaler`` is used only on CUDA, where fp16 loss scaling is
required; CPU bfloat16 needs no scaler. Numerics under AMP are close to, but
not bit-identical to, fp32.
"""

import contextlib
from typing import Any, Optional

import torch


def _probe(device_type: str, dtype: torch.dtype) -> bool:
    """Return True when a tiny autocast op runs for this dtype/device."""
    try:
        probe = torch.ones(2, 2, device=device_type)
    except Exception:
        return False
    try:
        with torch.autocast(device_type, dtype=dtype):
            (probe @ probe).sum()
    except Exception:
        return False
    return True


def _autocast_dtype(
    enabled: bool, device: torch.device
) -> Optional[torch.dtype]:
    """Return the autocast dtype to use, or None to stay in fp32."""
    if not enabled:
        return None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return None
        candidate = torch.float16
    else:
        candidate = torch.bfloat16
    return candidate if _probe(device.type, candidate) else None


class AmpController:
    """Resolve and apply an opt-in autocast/GradScaler AMP policy."""

    def __init__(self, enabled: bool, device: torch.device) -> None:
        """Resolve the autocast dtype and, on CUDA, build a GradScaler."""
        self._device = device
        self._requested = bool(enabled)
        self._dtype = _autocast_dtype(self._requested, device)
        self._scaler = self._make_scaler()

    def _make_scaler(self) -> Optional[torch.amp.GradScaler]:
        """Return a CUDA GradScaler for fp16, else None (no CPU scaling)."""
        if self._dtype is None or self._device.type != "cuda":
            return None
        return torch.amp.GradScaler(self._device.type)

    @property
    def requested(self) -> bool:
        """Return True when AMP was asked for, even if it fell back."""
        return self._requested

    @property
    def enabled(self) -> bool:
        """Return True when autocast actually runs."""
        return self._dtype is not None

    @property
    def dtype(self) -> Optional[str]:
        """Return the autocast dtype name ('float16'/'bfloat16'), or None."""
        if self._dtype is None:
            return None
        return repr(self._dtype).split(".")[-1]

    @property
    def scaling(self) -> bool:
        """Return True when a CUDA GradScaler is active."""
        return self._scaler is not None

    def autocast(self) -> Any:
        """Return the autocast context, or a no-op when AMP is inactive."""
        if self._dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self._device.type, dtype=self._dtype)

    def backward(self, loss: torch.Tensor, optimizer: Any) -> None:
        """Zero grads, backprop (scaled on CUDA), and step the optimiser."""
        optimizer.zero_grad()
        if self._scaler is None:
            loss.backward()
            optimizer.step()
            return
        self._scaler.scale(loss).backward()
        self._scaler.step(optimizer)
        self._scaler.update()
