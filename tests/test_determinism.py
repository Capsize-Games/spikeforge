"""Deterministic mode toggling, restoration, and the fixture check."""

from typing import Any, Iterator

import pytest
import torch

from snn_interpreter.network import model_store
from snn_interpreter.tracking.determinism import (
    bit_exactness_check,
    disable_deterministic,
    enable_deterministic,
)
from snn_interpreter.tracking.seed import set_seed
from snn_interpreter.training.training_engine import TrainingEngine


@pytest.fixture(autouse=True)
def _restore_flags() -> Iterator[None]:
    """Restore the torch backend flags after a test."""
    yield
    disable_deterministic()


def test_enable_sets_flags_and_reports() -> None:
    """Deterministic mode turns the flags on and reports what it applied."""
    report = enable_deterministic(123)
    payload = report.to_dict()
    assert payload["enabled"] is True
    assert payload["seed"] == 123
    assert payload["applied"]["torch"] is True
    assert payload["applied"]["cudnn_deterministic"] is True
    assert payload["notes"]
    assert torch.are_deterministic_algorithms_enabled() is True


def test_disable_restores_prior_flags() -> None:
    """Disabling returns the process to its pre-existing backend flags."""
    before = torch.are_deterministic_algorithms_enabled()
    enable_deterministic(1)
    assert torch.are_deterministic_algorithms_enabled() is True
    disable_deterministic()
    assert torch.are_deterministic_algorithms_enabled() is before
    assert not torch.backends.cudnn.deterministic


def test_bit_exactness_check_matches_and_differs() -> None:
    """The fixture check reports exact matches and honest differences."""
    def run() -> torch.Tensor:
        set_seed(7)
        return torch.randn(8)

    result = bit_exactness_check(run, run)
    assert result["exact"] is True
    assert result["max_abs_diff"] == 0.0
    other = bit_exactness_check(run, lambda: torch.zeros(8))
    assert other["exact"] is False
    assert other["max_abs_diff"] > 0.0


def test_engine_records_determinism_block(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deterministic engine records a determinism block in its manifest."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))
    engine = TrainingEngine(
        dataset="mnist", num_steps=2, device="cpu", subset=2, epochs=1,
        seed=11, deterministic=True,
    )
    engine.save("det")
    stored = model_store.manifest("det")
    assert stored["determinism"]["enabled"] is True
    assert stored["determinism"]["seed"] == 11
    assert stored["determinism"]["notes"]
