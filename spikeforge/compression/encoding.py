"""State-dict-level weight compression built on ``codec``'s tensor codec.

``compress_state_dict``/``dequantize_state_dict`` are the public entry
points: they quantize every floating tensor in a state dict, pass every
other entry through unchanged, and record a manifest-ready ``encoding``
block with the byte totals and quantize round-trip error.
"""

from typing import Any, Dict, Mapping, Tuple

import torch

from spikeforge.compression.codec import (
    SCHEME_INT8,
    SCHEME_NONE,
    WEIGHTS_ENCODING_VERSION,
    CompressedWeights,
    dequantize_tensor,
    quantize_tensor,
    resolve_scheme,
)
from spikeforge.compression.errors import CompressionError
from spikeforge.compression.quantize_stats import QuantizeStats


def _quantize_and_record(
    name: str,
    tensor: torch.Tensor,
    resolved: str,
    tensors: Dict[str, torch.Tensor],
    entries: Dict[str, Any],
) -> Tuple[int, int, torch.Tensor, Dict[str, Any]]:
    """Quantize ``tensor``, record it, and return its byte counts."""
    original_bytes = int(tensor.numel()) * int(tensor.element_size())
    codes, meta = quantize_tensor(tensor, resolved)
    tensors[name] = codes
    entries[name] = meta
    compressed_bytes = int(codes.numel()) * int(codes.element_size())
    return original_bytes, compressed_bytes, codes, meta


def _accumulate_entry(
    name: str,
    value: Any,
    resolved: str,
    tensors: Dict[str, torch.Tensor],
    entries: Dict[str, Any],
) -> Tuple[int, int, int, float, float]:
    """Quantize or pass through one entry; return its stat deltas."""
    tensor = value if torch.is_tensor(value) else None
    if tensor is None or not tensor.is_floating_point():
        tensors[name] = value
        return 0, 0, 0, 0.0, 0.0
    orig, comp, codes, meta = _quantize_and_record(
        name, tensor, resolved, tensors, entries
    )
    diff = (dequantize_tensor(codes, meta) - tensor.float()).abs()
    count = int(tensor.numel())
    if not diff.numel():
        return orig, comp, count, 0.0, 0.0
    return orig, comp, count, float(diff.max()), float(diff.sum())


def _quantize_entries(
    state_dict: Mapping[str, Any], resolved: str
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], QuantizeStats]:
    """Quantize every floating tensor; pass the rest through unchanged."""
    tensors: Dict[str, torch.Tensor] = {}
    entries: Dict[str, Any] = {}
    orig_bytes = comp_bytes = elements = 0
    err_peak = err_sum = 0.0
    for name, value in state_dict.items():
        orig, comp, count, peak, err = _accumulate_entry(
            name, value, resolved, tensors, entries
        )
        orig_bytes += orig
        comp_bytes += comp
        elements += count
        err_peak = max(err_peak, peak)
        err_sum += err
    stats = QuantizeStats(orig_bytes, comp_bytes, elements, err_peak, err_sum)
    return tensors, entries, stats


def _peak_magnitude(state_dict: Mapping[str, Any]) -> float:
    """Return the largest absolute value among floating tensors."""
    return float(
        max(
            (
                tensor.abs().max()
                for tensor in state_dict.values()
                if torch.is_tensor(tensor)
                and tensor.is_floating_point()
                and tensor.numel()
            ),
            default=0.0,
        )
    )


def _byte_error_fields(stats: QuantizeStats, peak: float) -> Dict[str, Any]:
    """Return the byte-count, ratio, and quantize-error fields."""
    relative = stats.err_peak / peak if peak > 0.0 else 0.0
    mean = stats.err_sum / stats.elements if stats.elements else 0.0
    return {
        "bytes": {
            "original": stats.orig_bytes,
            "compressed": stats.comp_bytes,
        },
        "ratio": (
            stats.orig_bytes / stats.comp_bytes if stats.comp_bytes else 0.0
        ),
        "error": {
            "max_abs": stats.err_peak,
            "mean_abs": mean,
            "relative": relative,
        },
    }


def _build_encoding(
    resolved: str,
    bits: int,
    state_dict: Mapping[str, Any],
    entries: Dict[str, Any],
    stats: QuantizeStats,
) -> Dict[str, Any]:
    """Assemble the manifest ``encoding`` block from quantize stats."""
    fields = {
        "version": WEIGHTS_ENCODING_VERSION,
        "scheme": resolved,
        "bits": int(bits),
        "tensors": entries,
        "counts": {"tensors": len(entries), "elements": stats.elements},
    }
    fields.update(_byte_error_fields(stats, _peak_magnitude(state_dict)))
    return fields


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
    resolved = resolve_scheme(scheme, bits)
    if resolved == SCHEME_NONE:
        raise CompressionError("the 'none' scheme produces no compression")
    tensors, entries, stats = _quantize_entries(state_dict, resolved)
    encoding = _build_encoding(resolved, bits, state_dict, entries, stats)
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
