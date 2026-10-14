"""Observed calibration ranges for fixed-point activation grids.

Split out of the ``activation_quant`` package root so each scheme class
lives in its own file. A :class:`Calibration` maps the stage keys the
quantizer uses to ``(min, max)`` pairs folded from a calibration pass, so
the grid is stable from the first served step.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, TypeGuard

import torch

from spikeforge_targets.activation_quant.scheme import TARGET_BOTH
from spikeforge_targets.activation_quant_records import tensor_range
from spikeforge_targets.fixed_point import EPS

#: A running (min, max) per calibration key.
Ranges = Dict[str, Tuple[float, float]]


def is_float(value: Any) -> TypeGuard[torch.Tensor]:
    """Return True when ``value`` is a floating torch tensor.

    Declared as a :class:`TypeGuard` so a caller holding an optional or
    duck-typed value is narrowed to ``Tensor`` by the check itself, rather
    than asserting the narrowing separately.
    """
    return torch.is_tensor(value) and value.is_floating_point()


@dataclass(frozen=True)
class Calibration:
    """Per-stage ranges observed on a calibration dataset.

    ``ranges`` maps the same stage keys the quantizer uses to an ``(min, max)``
    pair, and ``samples`` records how many tensors were folded in. A quantizer
    handed a calibration uses these bounds in place of its per-step extremes,
    so the grid is stable from the first served step. ``source`` names where
    the ranges came from, so a report can say whether the grid was chosen
    from a separate dataset or from the very fixture it is checked on.
    """

    bits: int
    target: str
    ranges: Mapping[str, Tuple[float, float]]
    samples: int = 0
    source: str = ""

    def bound(self, key: str) -> Optional[float]:
        """Return the symmetric magnitude for ``key``, or ``None``."""
        span = self.ranges.get(key)
        if span is None:
            return None
        return max(abs(float(span[0])), abs(float(span[1])))

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the calibration."""
        return {
            "bits": self.bits,
            "target": self.target,
            "samples": self.samples,
            "source": self.source,
            "ranges": {
                key: [float(span[0]), float(span[1])]
                for key, span in self.ranges.items()
            },
        }


def fold_range(seen: Ranges, key: str, value: Any) -> bool:
    """Fold ``value``'s range into ``seen`` under ``key``.

    Returns True when a floating tensor was folded and False when the value
    was ignored, so a caller can count samples.
    """
    if not is_float(value):
        return False
    low, high = tensor_range(torch.as_tensor(value).float())
    earlier = seen.get(key)
    if earlier is None:
        seen[key] = (low, high)
    else:
        seen[key] = (min(earlier[0], low), max(earlier[1], high))
    return True


def calibrate(
    tensors: Iterable[Tuple[str, Any]],
    *,
    bits: int = 8,
    target: str = TARGET_BOTH,
    source: str = "",
) -> Calibration:
    """Return per-key ranges folded from an observed tensor iterable.

    ``tensors`` yields ``(key, tensor)`` pairs - typically the stage outputs
    and carried-state tensors of a short calibration pass. Non-floating
    values are ignored; the ranges are the running min/max per key.
    """
    seen: Ranges = {}
    samples = 0
    for key, value in tensors:
        if fold_range(seen, key, value):
            samples += 1
    return Calibration(int(bits), str(target), seen, samples, str(source))


def calibrated_bound(
    calibration: Optional[Calibration], key: str, tensor: torch.Tensor
) -> float:
    """Return the calibration bound for ``key``, or the tensor's own peak."""
    if calibration is not None:
        bound = calibration.bound(key)
        if bound is not None and bound > EPS:
            return bound
    return float(tensor.abs().max()) if tensor.numel() else 0.0
