"""_input_features()/_dummy_spikes() resolve the right param per preset.

Regression test for a real bug: ``sequence_mlp`` has no ``input_size``
param at all (it names the same thing ``features``), so
``_input_features()`` silently fell through to the historical
MNIST-shaped ``28 * 28`` default. Invisible on CPU, since
``device.warmup()`` no-ops there -- only a CUDA run actually forwards
the dummy tensor through the network and hits the shape mismatch.
"""

from typing import Any

from spikeforge.training.training_engine import TrainingEngine


def _engine(**kwargs: Any) -> TrainingEngine:
    """Return a CPU engine with small, download-free defaults."""
    defaults: dict = {"dataset": "mnist", "num_steps": 4, "device": "cpu"}
    defaults.update(kwargs)
    return TrainingEngine(**defaults)


def test_input_size_presets_keep_their_existing_feature_count() -> None:
    """fc_legacy's own input_size still resolves unchanged."""
    engine = _engine(topology="fc_legacy", topology_params={"input_size": 64})
    assert engine._input_features() == 64


def test_sequence_mlp_resolves_features_not_the_mnist_default() -> None:
    """sequence_mlp's `features` param is used, not the 28*28 fallback."""
    engine = _engine(
        topology="sequence_mlp", topology_params={"features": 25},
    )
    assert engine._input_features() == 25


def test_dummy_spikes_matches_sequence_mlp_features() -> None:
    """The CUDA-warmup dummy tensor is shaped for the real features count."""
    engine = _engine(
        topology="sequence_mlp", topology_params={"features": 25},
    )
    assert engine._dummy_spikes().shape[-1] == 25
