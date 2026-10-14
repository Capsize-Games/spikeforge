"""The torch codec and the numpy schemes share one grid module."""

import torch

from spikeforge.compression.codec import dequantize_tensor, quantize_tensor
from spikeforge_targets.quantize_schemes import int8_codes, uint8_codes


def test_codec_and_simulation_share_grids() -> None:
    """The torch codec and the numpy schemes emit identical grids."""
    torch.manual_seed(0)
    values = torch.randn(4, 8)
    array = values.numpy()
    codes, meta = quantize_tensor(values, "int8")
    expected, grid = int8_codes(array)
    assert meta["scale"] == grid.scale
    assert meta["zero_point"] == grid.zero_point
    assert codes.numpy().tolist() == expected.tolist()
    codes, meta = quantize_tensor(values, "uint8")
    expected, grid = uint8_codes(array)
    assert meta["scale"] == grid.scale
    assert meta["zero_point"] == grid.zero_point
    assert codes.numpy().tolist() == expected.tolist()


def test_codec_preserves_constant_layers() -> None:
    """The shared degenerate grids reproduce constants instead of erasing."""
    for level in (2.5, -2.5, 0.0):
        codes, meta = quantize_tensor(
            torch.full((2, 3), level), "uint8"
        )
        assert meta["scale"] > 0.0
        assert torch.equal(
            dequantize_tensor(codes, meta), torch.full((2, 3), level)
        )
    codes, meta = quantize_tensor(torch.zeros(2, 3), "int8")
    assert meta["scale"] == 1.0
    assert torch.equal(dequantize_tensor(codes, meta), torch.zeros(2, 3))
