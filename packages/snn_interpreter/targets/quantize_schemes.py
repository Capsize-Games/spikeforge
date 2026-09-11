"""Per-tensor weight quantization schemes a target can declare.

Each scheme maps a float weight tensor to a quantize-and-dequantize
approximation that keeps the source dtype but is restricted to a finite set
of levels, and returns the before/after value ranges so a report can quantify
the clamping. The scheme names match the constraint strings declared in the
target catalog (``none``, ``weight_int8``, ``weight_uint8``).
"""

from typing import Callable, Dict, Tuple

import numpy as np

#: A quantized weight, its source range, and its quantized range.
Quantized = Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]
#: One scheme: a pure float-array-to-:data:`Quantized` function.
Scheme = Callable[[np.ndarray], Quantized]

#: Levels spanned by an unsigned 8-bit weight grid.
UINT8_LEVELS = 255.0
#: Peak below which a tensor is treated as all-zero and left untouched.
_EPS = 1e-12


def _range(weights: np.ndarray) -> Tuple[float, float]:
    """Return the (min, max) value range of a weight tensor."""
    if weights.size == 0:
        return (0.0, 0.0)
    return (float(weights.min()), float(weights.max()))


def none(weights: np.ndarray) -> Quantized:
    """Return ``weights`` unchanged; the honest no-op scheme."""
    value = np.asarray(weights, dtype=np.float32)
    span = _range(value)
    return value, span, span


def weight_int8(weights: np.ndarray) -> Quantized:
    """Symmetric per-tensor int8 quantize-dequantize of ``weights``."""
    value = np.asarray(weights, dtype=np.float32)
    peak = float(np.max(np.abs(value))) if value.size else 0.0
    if peak <= _EPS:
        return np.zeros_like(value), _range(value), (0.0, 0.0)
    scale = peak / 127.0
    code = np.clip(np.round(value / scale), -128.0, 127.0)
    quantized = (code * scale).astype(np.float32)
    return quantized, _range(value), _range(quantized)


def weight_uint8(weights: np.ndarray) -> Quantized:
    """Asymmetric per-tensor uint8 quantize-dequantize of ``weights``."""
    value = np.asarray(weights, dtype=np.float32)
    low, high = _range(value)
    if high - low <= _EPS:
        return np.zeros_like(value), (low, high), (0.0, 0.0)
    scale = (high - low) / UINT8_LEVELS
    zero = float(np.round(-low / scale))
    code = np.clip(np.round(value / scale) + zero, 0.0, UINT8_LEVELS)
    quantized = ((code - zero) * scale).astype(np.float32)
    return quantized, (low, high), _range(quantized)


#: The executable scheme for each declared ``quantization`` constraint name.
SCHEMES: Dict[str, Scheme] = {
    "none": none,
    "weight_int8": weight_int8,
    "weight_uint8": weight_uint8,
}
