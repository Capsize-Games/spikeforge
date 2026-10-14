"""The shared quantization grids both producers derive from."""

import math

from spikeforge.compression.grid import (
    EPS,
    INT8_MAX_CODE,
    INT8_MIN_CODE,
    UINT8_LEVELS,
    UINT8_MAX_CODE,
    UINT8_MIN_CODE,
    int8_grid,
    is_degenerate_int8,
    is_degenerate_uint8,
    uint8_degenerate,
    uint8_grid,
)


def test_code_bounds_pin_the_documented_ranges() -> None:
    """The symmetric grid never emits -128; the unsigned grid spans 256."""
    assert (INT8_MIN_CODE, INT8_MAX_CODE) == (-127, 127)
    assert (UINT8_MIN_CODE, UINT8_MAX_CODE) == (0, 255)
    assert UINT8_LEVELS == 255.0
    assert EPS == 1e-12


def test_int8_grid_maps_the_peak_onto_code_127() -> None:
    """A live tensor's scale is its peak over 127 with a zero point."""
    grid = int8_grid(2.0)
    assert grid.scale == 2.0 / 127.0
    assert grid.zero_point == 0
    assert is_degenerate_int8(2.0) is False


def test_int8_degenerate_grid_is_loadable_and_zero() -> None:
    """An all-zero tensor maps to (1.0, 0): zero codes decode to zeros."""
    for peak in (0.0, 1e-13):
        assert is_degenerate_int8(peak) is True
        grid = int8_grid(peak)
        assert grid.scale == 1.0
        assert grid.zero_point == 0
        assert math.isfinite(grid.scale) and grid.scale > 0.0
        assert (0 - grid.zero_point) * grid.scale == 0.0


def test_uint8_grid_spans_the_observed_range() -> None:
    """A live range maps onto 256 levels with a shifting zero-point."""
    grid = uint8_grid(-2.0, 6.0)
    assert grid.scale == 8.0 / 255.0
    assert grid.zero_point == 64
    assert is_degenerate_uint8(-2.0, 6.0) is False


def test_uint8_zero_point_is_unclamped() -> None:
    """Single-sign tensors keep an exact grid instead of clipping flat."""
    grid = uint8_grid(1.0, 2.0)
    assert grid.scale == 1.0 / 255.0
    assert grid.zero_point == -255


def test_uint8_degenerate_grid_reproduces_the_level() -> None:
    """A constant tensor decodes to its level with integer codes."""
    grid, code = uint8_degenerate(2.5, 2.5)
    assert (grid.scale, grid.zero_point, code) == (2.5, 0, 1)
    grid, code = uint8_degenerate(-2.5, -2.5)
    assert (grid.scale, grid.zero_point, code) == (2.5, 1, 0)
    grid, code = uint8_degenerate(0.0, 0.0)
    assert (grid.scale, grid.zero_point, code) == (1.0, 0, 0)
    for level in (2.5, -2.5, 0.0):
        grid, code = uint8_degenerate(level, level)
        assert math.isfinite(grid.scale) and grid.scale > 0.0
        assert (code - grid.zero_point) * grid.scale == level


def test_uint8_degenerate_grid_covers_near_constant() -> None:
    """A range narrower than EPS reproduces its midpoint, not NaN."""
    low, high = 1.0, 1.0 + 5e-13
    assert is_degenerate_uint8(low, high) is True
    grid = uint8_grid(low, high)
    assert math.isfinite(grid.scale) and grid.scale > 0.0
    assert grid.scale == (low + high) / 2.0
    assert grid.zero_point == 0
