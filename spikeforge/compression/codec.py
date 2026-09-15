"""Deterministic 8-bit weight compression for deployment bundles.

A float weight tensor is mapped to integer codes plus one ``(scale,
zero_point)`` pair, so the payload is one byte per live element instead of
four. Two schemes are supported and both round-trip through the exact same
dequantize rule ``(code - zero_point) * scale``:

``int8``   symmetric per-tensor; codes are ``int8`` in ``[-127, 127]`` and
           ``zero_point`` is always zero.
``uint8``  asymmetric per-tensor; codes are ``uint8`` in ``[0, 255]`` and the
           zero-point shifts the observed range onto the unsigned grid.

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

#: Largest magnitude a symmetric int8 code may take.
_INT8_LEVELS = 127.0
#: Span of the unsigned 8-bit grid.
_UINT8_LEVELS = 255.0
#: Peak below which a tensor is treated as all-zero.
_EPS = 1e-12


def _int8_codes(values: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    """Return symmetric int8 codes, scale, and zero-point for ``values``."""
    peak = float(values.abs().max()) if values.numel() else 0.0
    if peak <= _EPS:
        return torch.zeros_like(values, dtype=torch.int8), 0.0, 0
    scale = peak / _INT8_LEVELS
    codes = torch.round(values / scale).clamp(-_INT8_LEVELS, _INT8_LEVELS)
    return codes.to(torch.int8), scale, 0


def _uint8_codes(values: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    """Return asymmetric uint8 codes, scale, zero-point for ``values``."""
    if values.numel() == 0:
        return torch.zeros_like(values, dtype=torch.uint8), 0.0, 0
    low = float(values.min())
    high = float(values.max())
    if high - low <= _EPS:
        return torch.zeros_like(values, dtype=torch.uint8), 0.0, 0
    scale = (high - low) / _UINT8_LEVELS
    zero = max(0, min(255, int(round(-low / scale))))
    codes = torch.round(values / scale) + zero
    codes = codes.clamp(0.0, _UINT8_LEVELS).to(torch.uint8)
    return codes, scale, zero


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
