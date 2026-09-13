"""Contract tests for the SNN-native affect regulator."""

import pytest
import torch

from spikeforge.affect import AffectiveRegulator


def test_regulator_returns_spikes_and_persistent_broadcast() -> None:
    """Expose spikes while carrying a continuous state between steps."""
    regulator = AffectiveRegulator(2, state_size=3, beta=0.9)
    first = regulator.step(torch.ones(1, 2))
    second = regulator.step(torch.zeros(1, 2))

    assert first.spikes.shape == (1, 3)
    assert first.broadcast.shape == (1, 3)
    assert torch.any(second.broadcast != 0)


def test_reset_clears_state_and_batch_size_is_supported() -> None:
    """Reset state cleanly and reallocate it for a changed batch size."""
    regulator = AffectiveRegulator(2)
    regulator.step(torch.ones(1, 2))
    regulator.reset_state(batch_size=2)
    output = regulator.step(torch.zeros(2, 2))

    assert output.broadcast.shape == (2, 2)
    assert torch.equal(output.broadcast, torch.zeros(2, 2))


@pytest.mark.parametrize("kwargs", [{"input_size": 0}, {"state_size": 0}])
def test_rejects_empty_dimensions(kwargs: dict[str, int]) -> None:
    """Reject dimensions that cannot represent a regulator."""
    kwargs.setdefault("input_size", 2)
    with pytest.raises(ValueError, match="must be positive"):
        AffectiveRegulator(**kwargs)


def test_rejects_wrong_input_shape() -> None:
    """Reject inputs that do not match the configured feature width."""
    regulator = AffectiveRegulator(2)

    with pytest.raises(ValueError, match="shape"):
        regulator.step(torch.zeros(2))
