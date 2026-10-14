"""Per-tensor 8-bit quantization grids shared by every producer.

The bundle codec (:mod:`spikeforge.compression.codec`), the target drift
simulation (:mod:`spikeforge_targets.quantize_schemes`), and the quantized
summary envelopes all derive their ``(scale, zero_point)`` grids from this
module, so the three can never disagree about the math. Every grid decodes
through the single rule ``(code - zero_point) * scale``:

``int8``   symmetric per-tensor; ``zero_point`` is always zero and the scale
           maps the largest magnitude onto code 127.
``uint8``  asymmetric per-tensor; the scale spans the observed range over
           256 levels and the zero-point shifts it onto the unsigned grid.

Degenerate tensors (all-zero for ``int8``, constant for ``uint8``) still get
a loadable grid: the scale is always positive and finite, matching the C
loader's ``scale > 0`` requirement, and the grid reproduces the degenerate
values exactly (zeros for ``int8``, the constant level for ``uint8``).

This module is stdlib-only so the drift simulation and the envelope producer
can share it without importing torch.
"""

from dataclasses import dataclass
from typing import Tuple

#: Levels a symmetric int8 grid spans on each side of zero.
INT8_LEVELS = 127.0
#: Levels an asymmetric uint8 grid spans across the observed range.
UINT8_LEVELS = 255.0
#: Span at or below which a tensor is degenerate (all-zero / constant).
EPS = 1e-12
#: Smallest code a symmetric int8 grid produces (-128 never occurs).
INT8_MIN_CODE = -127
#: Largest code a symmetric int8 grid produces.
INT8_MAX_CODE = 127
#: Smallest code an asymmetric uint8 grid produces.
UINT8_MIN_CODE = 0
#: Largest code an asymmetric uint8 grid produces.
UINT8_MAX_CODE = 255


@dataclass(frozen=True)
class Grid:
    """One ``(scale, zero_point)`` pair decoding integer codes to floats."""

    scale: float
    zero_point: int


def is_degenerate_int8(peak: float) -> bool:
    """Return True when ``peak`` is too small for a symmetric grid."""
    return peak <= EPS


def is_degenerate_uint8(low: float, high: float) -> bool:
    """Return True when ``[low, high]`` is too narrow for a uint8 grid."""
    return high - low <= EPS


def int8_grid(peak: float) -> Grid:
    """Return the symmetric grid for a tensor peaking at ``peak``.

    A degenerate tensor maps to ``(1.0, 0)``: its codes are all zero, so
    it decodes to zeros exactly while keeping the scale C-loadable.
    """
    if is_degenerate_int8(peak):
        return Grid(1.0, 0)
    return Grid(peak / INT8_LEVELS, 0)


def uint8_grid(low: float, high: float) -> Grid:
    """Return the asymmetric grid spanning ``[low, high]``.

    A degenerate range maps to the single-level grid reproducing its
    midpoint (see :func:`uint8_degenerate`); a live range maps onto the
    full unsigned span with an unclamped zero-point, so single-sign
    tensors keep their accuracy instead of clipping onto one level.
    """
    if is_degenerate_uint8(low, high):
        return uint8_degenerate(low, high)[0]
    scale = (high - low) / UINT8_LEVELS
    return Grid(scale, int(round(-low / scale)))


def uint8_degenerate(low: float, high: float) -> Tuple[Grid, int]:
    """Return the grid and fill code reproducing a degenerate range.

    The midpoint of ``[low, high]`` is the reproduced level: it equals
    every value when the tensor is exactly constant and sits within
    ``EPS`` of every value when it is only near-constant. A positive
    level uses ``(level, 0)`` with code 1, a negative level uses
    ``(-level, 1)`` with code 0, and zero uses ``(1.0, 0)`` with code 0,
    so the level always decodes exactly with integer codes.
    """
    level = (low + high) / 2.0
    if level > 0.0:
        return Grid(level, 0), 1
    if level < 0.0:
        return Grid(-level, 1), 0
    return Grid(1.0, 0), 0
