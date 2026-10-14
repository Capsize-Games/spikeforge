"""Deterministic 8-bit weight compression for deployment bundles.

A float weight tensor is mapped to integer codes plus one ``(scale,
zero_point)`` pair, so the payload is one byte per live element instead of
four. Two schemes are supported and both round-trip through the exact same
dequantize rule ``(code - zero_point) * scale``:

``int8``   symmetric per-tensor; codes are ``int8`` in ``[-127, 127]`` and
           ``zero_point`` is always zero.
``uint8``  asymmetric per-tensor; codes are ``uint8`` in ``[0, 255]`` and the
           zero-point shifts the observed range onto the unsigned grid.

Grids come from :mod:`spikeforge.compression.grid`, shared with the target
drift simulation, so both agree on every ``(scale, zero_point)`` pair by
construction. Degenerate grids stay loadable (a positive scale) and
reproduce their values: an all-zero tensor decodes to zeros and a constant
tensor decodes to its level rather than being erased.

Only 8-bit schemes are implemented. A request for another bit-width is
refused with :class:`~spikeforge.compression.errors.CompressionError` rather
than rounded to 8-bit behind the caller's back. Non-floating tensors (integer
buffers, masks) are passed through unchanged and are never recorded as
compressed.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch

from spikeforge.compression.errors import CompressionError
from spikeforge.compression.grid import (
    INT8_MAX_CODE,
    INT8_MIN_CODE,
    UINT8_MAX_CODE,
    UINT8_MIN_CODE,
    int8_grid,
    is_degenerate_uint8,
    uint8_degenerate,
    uint8_grid,
)

#: Version of the weight-encoding contract recorded in a bundle manifest.
WEIGHTS_ENCODING_VERSION: int = 1
#: No compression: the raw float state dict is stored.
SCHEME_NONE = "none"
#: Symmetric per-tensor int8 quantization.
SCHEME_INT8 = "int8"
#: Asymmetric per-tensor uint8 quantization.
SCHEME_UINT8 = "uint8"
#: Every scheme this codec can execute.
SCHEMES: Tuple[str, ...] = (SCHEME_NONE, SCHEME_INT8, SCHEME_UINT8)


def _int8_codes(values: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    """Return symmetric int8 codes, scale, and zero-point for ``values``.

    The shared degenerate grid ``(1.0, 0)`` rounds near-zero values to
    code zero, so no separate all-zero branch is needed.
    """
    peak = float(values.abs().max()) if values.numel() else 0.0
    grid = int8_grid(peak)
    codes = torch.round(values / grid.scale).clamp(
        INT8_MIN_CODE, INT8_MAX_CODE
    )
    return codes.to(torch.int8), grid.scale, grid.zero_point


def _degenerate_uint8(
    values: torch.Tensor, low: float, high: float
) -> Tuple[torch.Tensor, float, int]:
    """Return the single-level codes, scale, zero-point for ``values``."""
    grid, code = uint8_degenerate(low, high)
    codes = torch.full(list(values.shape), code, dtype=torch.uint8)
    return codes, grid.scale, grid.zero_point


def _uint8_codes(values: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    """Return asymmetric uint8 codes, scale, zero-point for ``values``."""
    if values.numel() == 0:
        grid = uint8_grid(0.0, 0.0)
        codes = torch.zeros_like(values, dtype=torch.uint8)
        return codes, grid.scale, grid.zero_point
    low = float(values.min())
    high = float(values.max())
    if is_degenerate_uint8(low, high):
        return _degenerate_uint8(values, low, high)
    grid = uint8_grid(low, high)
    codes = torch.round(values / grid.scale) + grid.zero_point
    codes = codes.clamp(UINT8_MIN_CODE, UINT8_MAX_CODE).to(torch.uint8)
    return codes, grid.scale, grid.zero_point


def _codes_and_grid(
    values: torch.Tensor, scheme: str
) -> Tuple[torch.Tensor, float, int]:
    """Return the integer codes, scale, and zero-point for ``values``."""
    if scheme == SCHEME_INT8:
        return _int8_codes(values)
    return _uint8_codes(values)


def quantize_tensor(
    tensor: Any, scheme: str = SCHEME_INT8
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return ``tensor`` quantized to integer codes plus its metadata.

    ``tensor`` must be a floating tensor; the returned codes keep its shape
    and the metadata records ``scale``, ``zero_point``, ``dtype``, and
    ``shape`` so :func:`dequantize_tensor` is a pure function of the pair.
    """
    values = torch.as_tensor(tensor).detach().to(torch.float32)
    codes, scale, zero = _codes_and_grid(values, scheme)
    meta = {
        "scale": float(scale),
        "zero_point": int(zero),
        "dtype": str(codes.dtype).replace("torch.", ""),
        "shape": [int(size) for size in values.shape],
    }
    return codes, meta


def dequantize_tensor(codes: Any, meta: Mapping[str, Any]) -> torch.Tensor:
    """Return the float tensor ``codes`` approximates under ``meta``."""
    value = torch.as_tensor(codes).to(torch.float32)
    return (value - float(meta["zero_point"])) * float(meta["scale"])


@dataclass(frozen=True)
class CompressedWeights:
    """An encoded state dict plus the manifest block describing it.

    ``tensors`` holds the integer codes (or a passthrough tensor for any
    non-floating entry). ``encoding`` is the JSON-serialisable block written
    into a bundle manifest; feeding both to :func:`dequantize_state_dict`
    reconstructs the float state dict exactly.
    """

    scheme: str
    bits: int
    tensors: Mapping[str, torch.Tensor]
    encoding: Mapping[str, Any]

    def ratio(self) -> float:
        """Return ``original_bytes / compressed_bytes`` (0 when empty)."""
        sizes = self.encoding.get("bytes", {})
        compressed = float(sizes.get("compressed", 0.0))
        if compressed <= 0.0:
            return 0.0
        return float(sizes.get("original", 0.0)) / compressed


def resolve_scheme(scheme: str, bits: int) -> str:
    """Return a validated 8-bit scheme name or raise a typed error."""
    if scheme == SCHEME_NONE:
        return scheme
    if scheme not in (SCHEME_INT8, SCHEME_UINT8):
        raise CompressionError(
            f"unknown weight compression scheme {scheme!r}; expected one of "
            f"{tuple(SCHEMES)}"
        )
    if int(bits) != 8:
        raise CompressionError(
            f"scheme {scheme!r} only supports 8 bits, got {bits}"
        )
    return scheme
