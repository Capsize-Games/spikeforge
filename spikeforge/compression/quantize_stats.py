"""Byte and quantize-error totals accumulated while compressing weights."""

from typing import NamedTuple


class QuantizeStats(NamedTuple):
    """Byte and quantize-error totals accumulated while compressing."""

    orig_bytes: int
    comp_bytes: int
    elements: int
    err_peak: float
    err_sum: float
