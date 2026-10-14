"""Unit tests for the latency RC transfer curve (issue #117).

:func:`convert_to_time` maps an input intensity to a spike time. The
shipped formula is ``tau * log(data / (data - threshold))``, which goes
non-finite at and below the threshold; these tests pin that behavior so
a numerics change fails loudly instead of silently reshaping the
tutorial's latency curve.
"""

import torch

from spikeforge.encoding.latency_trainer import convert_to_time


def test_convert_to_time_matches_the_rc_formula() -> None:
    """Safe inputs follow ``tau * log(data / (data - threshold))``."""
    data = torch.tensor([0.5, 1.0, 2.0])
    expected = 5.0 * torch.log(data / (data - 0.01))
    assert torch.allclose(convert_to_time(data), expected)


def test_convert_to_time_default_parameters() -> None:
    """The defaults stay tau=5.0 and threshold=0.01."""
    data = torch.tensor([0.5, 1.0])
    assert torch.allclose(
        convert_to_time(data), convert_to_time(data, 5.0, 0.01)
    )


def test_convert_to_time_at_threshold_is_positive_infinity() -> None:
    """``data == threshold`` divides by zero, so the time is +inf."""
    result = convert_to_time(torch.tensor([0.01]), tau=5.0, threshold=0.01)
    assert torch.isinf(result).all()
    assert (result > 0).all()


def test_convert_to_time_below_threshold_is_nan() -> None:
    """``data < threshold`` takes the log of a negative, yielding nan."""
    result = convert_to_time(torch.tensor([0.005]), tau=5.0, threshold=0.01)
    assert torch.isnan(result).all()


def test_convert_to_time_at_zero_is_negative_infinity() -> None:
    """``data == 0`` takes the log of zero, yielding -inf."""
    result = convert_to_time(torch.tensor([0.0]), tau=5.0, threshold=0.01)
    assert torch.isinf(result).all()
    assert (result < 0).all()
