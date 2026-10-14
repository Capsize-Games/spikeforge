"""The reference-training script on the event training path.

The script has to pick the event engine by modality, one epoch has to
be a real pass over the training split rather than the dashboard's
batch cap, and the training split has to be visited shuffled.

The tonic backend is stubbed with a dataset of known, deliberately
unequal split lengths, so "complete" and "full" are assertions about
counts rather than about a flag being accepted.
"""

from pathlib import Path

import pytest
import torch
from _event_helpers import _TRAIN_SAMPLES, _FakeTonic, _reference

import scripts.train_reference_models as trm
from spikeforge.data import event_loader
from spikeforge.events import event_source
from spikeforge.network import model_store
from spikeforge.training.event_engine import EventTrainingEngine
from spikeforge.training.training_engine import TrainingEngine


@pytest.fixture
def tonic_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the event path through the stub instead of a real download."""
    monkeypatch.setattr(event_source, "dataset_available", lambda name: True)
    monkeypatch.setattr(
        event_loader.tonic_api, "dataset_class", lambda name: _FakeTonic
    )

@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep any checkpoint write inside the test's own directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


# --- engine selection ----------------------------------------------------


def test_modality_selects_the_engine_not_the_dataset_name(
    tonic_stub: None,
) -> None:
    """A new event dataset needs no change to the script to be routed."""
    assert trm._is_event("dvs128_gesture") is True
    assert trm._is_event("mnist") is False
    engine = trm._engine_for(_reference())
    assert isinstance(engine, EventTrainingEngine)


def test_image_rows_keep_the_historical_engine() -> None:
    """Routing by modality must not disturb the image path."""
    reference = _reference(
        dataset="mnist", topology="fc_small", topology_params={}
    )
    engine = trm._engine_for(reference)
    assert isinstance(engine, TrainingEngine)
    assert not isinstance(engine, EventTrainingEngine)


# --- a full training epoch ----------------------------------------------


def test_a_published_event_epoch_covers_the_whole_training_split(
    tonic_stub: None,
) -> None:
    """The dashboard's batch cap must not decide what a published row saw.

    Left at its default the epoch would be ``EPOCH_BATCHES * batch_size``
    samples whatever the split's size, while the entry's notes claim the full
    training split.
    """
    engine = trm._engine_for(_reference())
    assert trm._use_full_event_epoch(engine, "n_mnist") == _TRAIN_SAMPLES
    batches = engine._epoch_batches(train=True)
    assert sum(len(labels) for _inputs, labels in batches) == _TRAIN_SAMPLES


def test_the_epoch_extent_is_recorded_in_the_manifest(
    tonic_stub: None,
) -> None:
    """A run whose epoch length was overridden has to say so to reproduce."""
    engine = trm._engine_for(_reference())
    trm._use_full_event_epoch(engine, "n_mnist")
    assert engine._manifest_config()["epoch_samples"] == _TRAIN_SAMPLES


def test_image_rows_are_left_alone_by_the_epoch_helper() -> None:
    """The image loader already walks the split, so nothing is overridden."""
    engine = trm._engine_for(
        _reference(dataset="mnist", topology="fc_small", topology_params={})
    )
    assert trm._use_full_event_epoch(engine, "mnist") is None


# --- shuffling the training split ---------------------------------------


def test_the_training_split_is_visited_in_a_shuffled_order(
    tonic_stub: None,
) -> None:
    """Real event datasets ship grouped by class, so order matters.

    SSC's training split is 35 contiguous runs over 75,466 samples. Read in
    order, every batch is a single class and the network learns only to name
    whichever class it is currently being shown -- which scored 3.84% against
    a 2.86% chance baseline on 35 classes. The synthetic fixture hid this
    completely, because its labels are ``index % num_classes`` and therefore
    perfectly interleaved by accident.
    """
    engine = trm._engine_for(_reference())
    trm._use_full_event_epoch(engine, "n_mnist")
    batches = engine._epoch_batches(train=True)
    visited = [
        int(label) for _inputs, labels in batches for label in labels
    ]
    assert len(visited) == _TRAIN_SAMPLES
    assert visited != sorted(visited)


def test_the_held_out_split_is_not_shuffled(tonic_stub: None) -> None:
    """Scoring reads the split in order, as the image loader does."""
    engine = trm._engine_for(_reference())
    first = engine._epoch_batches(train=False)
    second = engine._epoch_batches(train=False)
    for (inputs, _l), (again, _r) in zip(first, second):
        assert torch.equal(inputs, again)


def test_the_shuffle_is_reproducible_from_the_seed(tonic_stub: None) -> None:
    """A published row has to be reproducible, so the order is seeded."""
    orders = []
    for _ in range(2):
        engine = trm._engine_for(_reference(seed=7))
        trm._use_full_event_epoch(engine, "n_mnist")
        orders.append([
            int(label)
            for _inputs, labels in engine._epoch_batches(train=True)
            for label in labels
        ])
    assert orders[0] == orders[1]


def test_a_different_seed_gives_a_different_order(tonic_stub: None) -> None:
    """The order really is seeded, not a fixed permutation."""
    def order(seed: int) -> list:
        engine = trm._engine_for(_reference(seed=seed))
        trm._use_full_event_epoch(engine, "n_mnist")
        return [
            int(label)
            for _inputs, labels in engine._epoch_batches(train=True)
            for label in labels
        ]

    assert order(1) != order(2)
