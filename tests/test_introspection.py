"""Hand-computed introspection metrics over a synthetic ``Trajectory``."""

import json

import pytest
import torch

from spikeforge.introspection.firing_rate import (
    firing_rate,
    stage_firing_rates,
)
from spikeforge.introspection.histogram import firing_rate_histogram
from spikeforge.introspection.isi import isi_stats
from spikeforge.introspection.metrics import trajectory_metrics
from spikeforge.introspection.sparsity import sparsity, stage_sparsity
from spikeforge.simulator.trajectory import Trajectory


def _spikes() -> torch.Tensor:
    """Two neurons over five steps: [1,0,1,0,1] and [0,1,0,0,1]."""
    return torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )


def test_firing_rate_is_mean_spikes_per_neuron_per_step() -> None:
    """Five spikes over ten entries is a rate of one half."""
    assert firing_rate(_spikes()) == pytest.approx(5 / 10)


def test_sparsity_is_fraction_of_zeros() -> None:
    """Half the entries are zero, so sparsity is one half."""
    assert sparsity(_spikes()) == pytest.approx(5 / 10)


def test_isi_statistics_are_hand_computable() -> None:
    """Intervals [2, 2] and [3] give mean 7/3 and population std."""
    stats = isi_stats(_spikes())
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(7 / 3)
    assert stats["median"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx((2 / 9) ** 0.5)
    assert stats["cv"] == pytest.approx((2 / 9) ** 0.5 / (7 / 3))


def test_isi_is_undefined_for_a_single_spike() -> None:
    """A neuron that spikes once has no interval to report."""
    stats = isi_stats(torch.tensor([[1.0], [0.0], [0.0]]))
    assert stats["count"] == 0
    assert stats["mean"] is None
    assert stats["cv"] is None


def test_histogram_counts_every_neuron() -> None:
    """A histogram over two neurons places both in the two bins."""
    histogram = firing_rate_histogram(_spikes(), bins=2)
    assert len(histogram["edges"]) == 3
    assert sum(histogram["counts"]) == 2


def test_trajectory_metrics_is_json_serialisable() -> None:
    """The aggregate metrics dict survives ``json.dumps`` unchanged."""
    trajectory = Trajectory(
        steps=5,
        logits=torch.zeros(1),
        spikes={"n": _spikes()},
    )
    metrics = trajectory_metrics(trajectory, bins=2)
    assert metrics["steps"] == 5
    stage = metrics["stages"]["n"]
    assert stage["firing_rate"] == pytest.approx(0.5)
    assert stage["sparsity"] == pytest.approx(0.5)
    assert stage["isi"]["count"] == 3
    assert isinstance(json.dumps(metrics), str)


def test_stage_helpers_cover_every_recorded_stage() -> None:
    """The per-stage helpers key their results by neuron stage name."""
    trajectory = Trajectory(
        steps=5, logits=torch.zeros(1), spikes={"n": _spikes()}
    )
    assert set(stage_firing_rates(trajectory)) == {"n"}
    assert set(stage_sparsity(trajectory)) == {"n"}
