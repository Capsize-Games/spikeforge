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


def _codes_and_grid(
    values: torch.Tensor, scheme: str
) -> Tuple[torch.Tensor, float, int]:
    """Return the integer codes, scale, and zero-point for ``values``."""
    if scheme == SCHEME_INT8:
        peak = float(values.abs().max()) if values.numel() else 0.0
        if peak <= _EPS:
            return torch.zeros_like(values, dtype=torch.int8), 0.0, 0
        scale = peak / _INT8_LEVELS
        codes = torch.round(values / scale).clamp(
            -_INT8_LEVELS, _INT8_LEVELS
        )
        return codes.to(torch.int8), scale, 0
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


def _resolve_scheme(scheme: str, bits: int) -> str:
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


def compress_state_dict(
    state_dict: Mapping[str, Any],
    scheme: str = SCHEME_INT8,
    bits: int = 8,
) -> CompressedWeights:
    """Return ``state_dict`` compressed under ``scheme`` and its encoding.

    Every floating parameter is quantized and every other entry is passed
    through verbatim. The encoding records the per-tensor grid, the byte
    totals (so a report can state a compression ratio), and the quantize
    round-trip error (so the accuracy cost is named rather than hidden).
    """
    resolved = _resolve_scheme(scheme, bits)
    if resolved == SCHEME_NONE:
        raise CompressionError("the 'none' scheme produces no compression")
    tensors: Dict[str, torch.Tensor] = {}
    entries: Dict[str, Any] = {}
    original_bytes = 0
    compressed_bytes = 0
    element_count = 0
    error_peak = 0.0
    error_sum = 0.0
    for name, value in state_dict.items():
        tensor = value if torch.is_tensor(value) else None
        if tensor is None or not tensor.is_floating_point():
            tensors[name] = value
            continue
        original_bytes += int(tensor.numel()) * int(tensor.element_size())
        codes, meta = quantize_tensor(tensor, resolved)
        tensors[name] = codes
        entries[name] = meta
        compressed_bytes += int(codes.numel()) * int(codes.element_size())
        element_count += int(tensor.numel())
        difference = (dequantize_tensor(codes, meta) - tensor.float()).abs()
        if difference.numel():
            error_peak = max(error_peak, float(difference.max()))
            error_sum += float(difference.sum())
    peak = float(
        max(
            (
                tensor.abs().max()
                for tensor in state_dict.values()
                if torch.is_tensor(tensor) and tensor.is_floating_point()
                and tensor.numel()
            ),
            default=0.0,
        )
    )
    relative = error_peak / peak if peak > 0.0 else 0.0
    mean = error_sum / element_count if element_count else 0.0
    encoding = {
        "version": WEIGHTS_ENCODING_VERSION,
        "scheme": resolved,
        "bits": int(bits),
        "tensors": entries,
        "counts": {"tensors": len(entries), "elements": element_count},
        "bytes": {
            "original": original_bytes,
            "compressed": compressed_bytes,
        },
        "ratio": (
            original_bytes / compressed_bytes if compressed_bytes else 0.0
        ),
        "error": {
            "max_abs": error_peak,
            "mean_abs": mean,
            "relative": relative,
        },
    }
    return CompressedWeights(resolved, int(bits), tensors, encoding)


def dequantize_state_dict(
    tensors: Mapping[str, Any], encoding: Mapping[str, Any]
) -> Dict[str, torch.Tensor]:
    """Return the float state dict ``tensors`` encodes under ``encoding``.

    An entry the encoding does not describe is returned unchanged, so a
    passthrough buffer survives a decompress/load cycle intact.
    """
    entries = encoding.get("tensors") or {}
    restored: Dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        meta = entries.get(name)
        if meta is None or not torch.is_tensor(value):
            restored[name] = value
            continue
        restored[name] = dequantize_tensor(value, meta)
    return restored
