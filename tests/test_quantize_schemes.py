"""Quantization must preserve nonzero weights in degenerate ranges."""

import numpy as np
import pytest

from spikeforge_targets.quantize_schemes import weight_uint8


@pytest.mark.parametrize("level", [2.5, -2.5, 0.0, 1e-13, -1e-13])
def test_uint8_preserves_constant_weights_and_ranges(level: float) -> None:
    """A constant tensor is already representable by one quantization level."""
    weights = np.full((2, 3), level, dtype=np.float32)
    original = weights.copy()
    quantized, before, after = weight_uint8(weights)
    np.testing.assert_array_equal(quantized, original)
    np.testing.assert_array_equal(weights, original)
    expected = (float(weights.min()), float(weights.max()))
    assert before == after == expected
    assert quantized is not weights


def test_uint8_preserves_near_constant_weights() -> None:
    """The tiny-range guard preserves values rather than erasing a layer."""
    weights = np.array([[1e-13, 2e-13]], dtype=np.float32)
    quantized, before, after = weight_uint8(weights)
    np.testing.assert_array_equal(quantized, weights)
    assert before == after


def test_uint8_empty_weights_have_zero_ranges() -> None:
    """Empty tensors stay empty without attempting a division by zero."""
    weights = np.empty((0, 2), dtype=np.float32)
    quantized, before, after = weight_uint8(weights)
    assert quantized.shape == weights.shape
    assert before == after == (0.0, 0.0)
