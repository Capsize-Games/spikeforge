"""Weight compression: deterministic codecs, size, and pruning reports."""

import json

import pytest
import torch

from spikeforge import compression
from spikeforge.compression import (
    SCHEME_INT8,
    SCHEME_UINT8,
    CompressionError,
    CompressionReport,
    compress_state_dict,
    dequantize_state_dict,
    dequantize_tensor,
    prune,
    quantize_tensor,
    sparsity,
)


def _state() -> dict:
    """Return a small mixed state dict with a non-float passthrough."""
    torch.manual_seed(0)
    return {
        "fc.weight": torch.randn(8, 16),
        "fc.bias": torch.randn(8),
        "steps": torch.tensor([1, 2, 3]),
    }


def test_int8_roundtrip_is_deterministic_and_bounded() -> None:
    """int8 codes round-trip within half a step and repeat exactly."""
    weight = torch.linspace(-1.0, 1.0, steps=257)
    codes, meta = quantize_tensor(weight, SCHEME_INT8)
    assert codes.dtype == torch.int8
    assert torch.equal(codes, quantize_tensor(weight, SCHEME_INT8)[0])
    restored = dequantize_tensor(codes, meta)
    scale = 1.0 / 127.0
    assert float((restored - weight).abs().max()) <= scale


def test_uint8_roundtrip_is_asymmetric_and_bounded() -> None:
    """uint8 codes land in the observed range and stay in bounds."""
    weight = torch.linspace(-2.0, 6.0, steps=511)
    codes, meta = quantize_tensor(weight, SCHEME_UINT8)
    assert codes.dtype == torch.uint8
    assert int(codes.min()) >= 0 and int(codes.max()) <= 255
    restored = dequantize_tensor(codes, meta)
    step = (6.0 - (-2.0)) / 255.0
    assert float((restored - weight).abs().max()) <= step


def test_state_dict_roundtrip_ratio_and_passthrough() -> None:
    """A compressed state dict reports a ratio > 1 and keeps its buffers."""
    state = _state()
    compressed = compress_state_dict(state, SCHEME_INT8)
    assert compressed.ratio() > 1.0
    assert compressed.encoding["bytes"]["compressed"] < (
        compressed.encoding["bytes"]["original"]
    )
    assert compressed.encoding["version"] == 1
    restored = dequantize_state_dict(compressed.tensors, compressed.encoding)
    assert torch.equal(restored["steps"], state["steps"])
    assert restored["steps"].dtype == torch.int64
    peak = float(state["fc.weight"].abs().max())
    assert float((restored["fc.weight"] - state["fc.weight"]).abs().max()) < (
        peak / 127.0
    )


def test_report_from_encoding_is_sane_and_json_serialisable() -> None:
    """The compression report restates scheme, counts, ratio, and error."""
    compressed = compress_state_dict(_state(), SCHEME_INT8)
    report = CompressionReport.from_encoding(compressed.encoding)
    payload = report.to_dict()
    assert json.dumps(payload)
    assert report.scheme == SCHEME_INT8
    assert report.bits == 8
    assert report.counts()["tensors"] == 2
    assert report.ratio > 1.0
    assert payload["error"]["max_abs"] >= 0.0


def test_unknown_scheme_and_bit_width_are_refused() -> None:
    """An unsupported scheme or bit-width is a named refusal, not a guess."""
    with pytest.raises(CompressionError):
        compress_state_dict(_state(), "int4")
    with pytest.raises(CompressionError):
        compress_state_dict(_state(), SCHEME_INT8, bits=4)
    with pytest.raises(CompressionError):
        compress_state_dict(_state(), "none")


def test_magnitude_pruning_reports_sparsity_and_drift() -> None:
    """Unstructured pruning zeroes at the target rate and reports drift."""
    state = _state()
    result = prune(state, 0.5)
    report = result.report
    assert report.strategy == "unstructured"
    assert 0.4 <= report.sparsity <= 1.0
    assert report.density() == pytest.approx(1.0 - report.sparsity)
    assert report.drift is not None
    assert report.drift["max_abs"] >= 0.0
    assert sparsity(result.tensors) == pytest.approx(report.sparsity)
    assert json.dumps(report.to_dict())


def test_structured_pruning_zeroes_whole_channels() -> None:
    """Structured pruning removes groups, not individual weights."""
    state = _state()
    result = prune(state, 0.5, strategy="structured")
    weight = result.tensors["fc.weight"]
    rows = weight.abs().sum(dim=1)
    assert int((rows == 0).sum()) >= 1
    assert result.report.strategy == "structured"


def test_pruning_rejects_out_of_range_and_unknown_strategy() -> None:
    """A bad sparsity level or strategy raises a typed error."""
    with pytest.raises(CompressionError):
        prune(_state(), 1.0)
    with pytest.raises(CompressionError):
        prune(_state(), -0.1)
    with pytest.raises(CompressionError):
        prune(_state(), 0.5, strategy="random")


def test_public_exports_match_the_scheme_registry() -> None:
    """The package exports the documented schemes and error type."""
    assert set(compression.SCHEMES) == {
        compression.SCHEME_NONE,
        compression.SCHEME_INT8,
        compression.SCHEME_UINT8,
    }
    assert issubclass(CompressionError, Exception)
