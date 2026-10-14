"""Smoke tests for the tutorial encoding trainers (issue #117).

The ``spikeforge-encodings`` demo wires three trainers
(:class:`DeltaTrainer`, :class:`LatencyTrainer`,
:class:`RandomSpikeGenerator`); these tests run each at tiny settings and
pin the shapes it produces. The MNIST-backed trainers download the
dataset on a cold cache, so they carry the ``network`` marker like the
other dataset tests.
"""

import pytest
import torch

from spikeforge.config import DATA_DIR
from spikeforge.encoding.delta_trainer import DeltaTrainer
from spikeforge.encoding.latency_trainer import LatencyTrainer
from spikeforge.encoding.random_spikegen import RandomSpikeGenerator
from spikeforge.training.trainer import SNNTrainer

#: Keys the latency trainer must encode, one per flag variant.
_LATENCY_KEYS = ("base", "linear", "normalized", "clip")


def _tiny_trainer() -> SNNTrainer:
    """Return a rate-coding trainer on a four-sample MNIST slice."""
    return SNNTrainer(
        data_path=DATA_DIR, subset=4, batch_size=2,
        vectorization_num_steps=2, do_shuffle=False,
    )


def _tiny_latency() -> LatencyTrainer:
    """Return a latency trainer on a four-sample MNIST slice."""
    # Steps stay at the tutorial default: the RC spike times of a real
    # batch reach ~58, and spikegen.latency rejects smaller horizons.
    return LatencyTrainer(
        latency_steps=100, data_path=DATA_DIR, subset=4, batch_size=2,
        vectorization_num_steps=2, do_shuffle=False,
    )


def test_delta_trainer_encodes_the_demo_series() -> None:
    """The default series yields on- and off-spike delta codes."""
    trainer = DeltaTrainer()
    assert trainer.data is not None
    assert trainer.spike_data is not None
    assert trainer.spike_data_off is not None
    assert trainer.spike_data.shape == trainer.data.shape
    assert trainer.spike_data_off.shape == trainer.data.shape


def test_delta_trainer_accepts_a_custom_series() -> None:
    """A custom time series is encoded instead of the demo one."""
    series = [0.0, 5.0, 0.0, -5.0, 0.0]
    trainer = DeltaTrainer(threshold=2, time_series=series)
    assert trainer.data is not None
    assert trainer.data.shape == (len(series),)
    assert trainer.spike_data is not None
    assert bool((trainer.spike_data != 0).any())


def test_random_spikegen_shape_matches_its_options() -> None:
    """The generated train has ``(num_steps, *size)`` binary values."""
    generator = RandomSpikeGenerator(num_steps=4, size=(8, 8))
    assert generator.spike_rand is not None
    assert generator.spike_rand.shape == (4, 8, 8)
    assert bool(((generator.spike_rand == 0)
                 | (generator.spike_rand == 1)).all())


def test_random_spikegen_defaults_to_hundred_steps() -> None:
    """The default generator keeps its 100-step 28x28 geometry."""
    generator = RandomSpikeGenerator()
    assert generator.spike_rand is not None
    assert generator.spike_rand.shape == (100, 28, 28)


@pytest.mark.network
def test_latency_trainer_encodes_every_variant() -> None:
    """All four latency flag variants encode the first batch."""
    trainer = _tiny_latency()
    assert trainer.latency_data is not None
    assert set(trainer.latency_data) == set(_LATENCY_KEYS)
    for spikes in trainer.latency_data.values():
        assert spikes.size(0) == 100
    assert trainer.latency_targets is not None
    assert trainer.latency_input is not None


@pytest.mark.network
def test_rate_trainer_produces_dual_gain_spikes() -> None:
    """The exporter precondition: unit- and low-gain spike data."""
    trainer = _tiny_trainer()
    assert trainer.spike_data is not None
    assert trainer.spike_data_low_gain is not None
    assert trainer.spike_data.shape == trainer.spike_data_low_gain.shape
    assert trainer.spike_data.size(0) == 2
    assert isinstance(trainer.gain, float)
    assert isinstance(trainer.interval, int)
    assert isinstance(trainer.data_size, tuple)
    assert torch.is_tensor(trainer.spike_targets)
