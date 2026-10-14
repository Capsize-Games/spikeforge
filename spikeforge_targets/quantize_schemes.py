"""Per-tensor weight quantization schemes a target can declare.

Each scheme maps a float weight tensor to a quantize-and-dequantize
approximation that keeps the source dtype but is restricted to a finite set
of levels, and returns the before/after value ranges so a report can quantify
the clamping. The scheme names match the constraint strings declared in the
target catalog (``none``, ``weight_int8``, ``weight_uint8``).

Grids come from :mod:`spikeforge.compression.grid`, shared with the bundle
codec, so the drift simulation and the stored compression agree on every
``(scale, zero_point)`` pair by construction. :func:`int8_codes` and
:func:`uint8_codes` additionally expose the integer codes behind the
approximation for the quantized summary envelopes.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from spikeforge.compression.grid import (
    INT8_MAX_CODE,
    INT8_MIN_CODE,
    UINT8_MAX_CODE,
    UINT8_MIN_CODE,
    Grid,
    int8_grid,
    is_degenerate_int8,
    is_degenerate_uint8,
    uint8_degenerate,
    uint8_grid,
)
from spikeforge.compression.grid import UINT8_LEVELS as UINT8_LEVELS

#: A quantized weight, its source range, and its quantized range.
Quantized = Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]
#: One scheme: a pure float-array-to-:data:`Quantized` function.
Scheme = Callable[[np.ndarray], Quantized]


def _range(weights: np.ndarray) -> Tuple[float, float]:
    """Return the (min, max) value range of a weight tensor."""
    if weights.size == 0:
        return (0.0, 0.0)
    return (float(weights.min()), float(weights.max()))


def _peak(weights: np.ndarray) -> float:
    """Return the largest magnitude in ``weights``, or 0.0 when empty."""
    if weights.size == 0:
        return 0.0
    return float(np.max(np.abs(weights)))


def node_weight(node: Any) -> Optional[np.ndarray]:
    """Return a node's floating weight array, or ``None`` when it has none.

    This is the single rule deciding which tensors a weight scheme (and
    the quantized envelopes) apply to: the ``weight`` attribute when it
    holds floating values. Biases and non-floating buffers are untouched.
    """
    value = getattr(node, "weight", None)
    if value is None:
        return None
    array = np.asarray(value)
    return array if array.dtype.kind == "f" else None


def int8_codes(weights: np.ndarray) -> Tuple[np.ndarray, Grid]:
    """Return symmetric int8 codes and their grid for ``weights``.

    Degenerate tensors map onto the shared ``(1.0, 0)`` grid with all-zero
    codes, decoding to zeros exactly with a C-loadable scale.
    """
    value = np.asarray(weights, dtype=np.float32)
    grid = int8_grid(_peak(value))
    codes = np.clip(
        np.round(value / grid.scale), INT8_MIN_CODE, INT8_MAX_CODE
    )
    return codes.astype(np.int8), grid


def uint8_codes(weights: np.ndarray) -> Tuple[np.ndarray, Grid]:
    """Return asymmetric uint8 codes and their grid for ``weights``.

    A degenerate range maps onto the shared single-level grid, so the
    codes decode to the constant level exactly with a C-loadable scale.
    """
    value = np.asarray(weights, dtype=np.float32)
    low, high = _range(value)
    if is_degenerate_uint8(low, high):
        grid, code = uint8_degenerate(low, high)
        return np.full(value.shape, code, dtype=np.uint8), grid
    grid = uint8_grid(low, high)
    codes = np.clip(
        np.round(value / grid.scale) + grid.zero_point,
        UINT8_MIN_CODE,
        UINT8_MAX_CODE,
    )
    return codes.astype(np.uint8), grid


def _decode(codes: np.ndarray, grid: Grid) -> np.ndarray:
    """Return ``codes`` dequantized to floats under ``grid``."""
    values = codes.astype(np.float32) - grid.zero_point
    return (values * grid.scale).astype(np.float32)


def none(weights: np.ndarray) -> Quantized:
    """Return ``weights`` unchanged; the honest no-op scheme."""
    value = np.asarray(weights, dtype=np.float32)
    span = _range(value)
    return value, span, span


def weight_int8(weights: np.ndarray) -> Quantized:
    """Symmetric per-tensor int8 quantize-dequantize of ``weights``."""
    value = np.asarray(weights, dtype=np.float32)
    if is_degenerate_int8(_peak(value)):
        return np.zeros_like(value), _range(value), (0.0, 0.0)
    codes, grid = int8_codes(value)
    quantized = _decode(codes, grid)
    return quantized, _range(value), _range(quantized)


def weight_uint8(weights: np.ndarray) -> Quantized:
    """Asymmetric per-tensor uint8 quantize-dequantize of ``weights``."""
    value = np.asarray(weights, dtype=np.float32)
    low, high = _range(value)
    if is_degenerate_uint8(low, high):
        return value.copy(), (low, high), (low, high)
    codes, grid = uint8_codes(value)
    quantized = _decode(codes, grid)
    return quantized, (low, high), _range(quantized)


#: The executable scheme for each declared ``quantization`` constraint name.
SCHEMES: Dict[str, Scheme] = {
    "none": none,
    "weight_int8": weight_int8,
    "weight_uint8": weight_uint8,
}
