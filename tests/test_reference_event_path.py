"""The reference-training script on the event path.

Three things had to change before an event row could mean what the image rows
mean, and each is checked here: the script has to pick the event engine by
modality, one epoch has to be a real pass over the training split rather than
the dashboard's batch cap, and the published accuracy has to come from the
*complete* held-out split rather than the four batches the dashboard scores.

The tonic backend is stubbed with a dataset of known, deliberately unequal
split lengths, so "complete" and "full" are assertions about counts rather than
about a flag being accepted.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch

import scripts.train_reference_models as trm
from spikeforge.data import event_loader
from spikeforge.events import event_source
from spikeforge.network import model_store
from spikeforge.training.event_engine import EventTrainingEngine
from spikeforge.training.training_engine import TrainingEngine

#: Split lengths of the stub. Unequal, and neither a multiple of the eval
#: batch, so an off-by-one or a truncation shows up as a wrong count.
_TRAIN_SAMPLES = 23
_TEST_SAMPLES = 7


class _FakeTonic:
    """A tonic-shaped 28x28 dataset with distinct train and test splits."""

    sensor_size = (28, 28, 2)

    def __init__(self, root: str, **kwargs: Any) -> None:
        """Record which split the registry's kwargs selected."""
        self._train = bool(kwargs.get("train"))

    def __len__(self) -> int:
        """Return this split's own length."""
        return _TRAIN_SAMPLES if self._train else _TEST_SAMPLES

    def __getitem__(self, index: int) -> Tuple[Dict[str, List[int]], int]:
        """Return a two-event stream whose column encodes the split."""
        column = index % 27
        offset = 0 if self._train else 1
        stream = {
            "x": [column, column + offset],
            "y": [index % 28, (index + 1) % 28],
            "t": [0, 1000],
            "p": [1, 0],
        }
        return stream, index % 4


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


def _reference(**overrides: Any) -> trm.Reference:
    """Return a tiny event reference configuration."""
    fields: Dict[str, Any] = {
        "name": "probe-event",
        "dataset": "n_mnist",
        "topology": "fc_legacy",
        "epochs": 1,
        "num_steps": 4,
        "hidden": 8,
        "batch_size": 4,
        "topology_params": {"input_size": 784},
    }
    fields.update(overrides)
    return trm.Reference(**fields)


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


def test_batch_helper_defaults_to_sequential() -> None:
    """The dashboard's path is unchanged: shuffling is opt-in."""
    from spikeforge.training.event_batches import visit_order

    assert visit_order(6, shuffle=False) == [0, 1, 2, 3, 4, 5]
    shuffled = visit_order(64, shuffle=True, seed=0)
    assert sorted(shuffled) == list(range(64))
    assert shuffled != list(range(64))


# --- the complete held-out split ----------------------------------------


def test_event_scoring_walks_the_complete_held_out_split(
    tonic_stub: None,
) -> None:
    """A published event accuracy covers every test sample, not four batches.

    ``EvalMixin`` caches ``EVAL_BATCHES`` batches for the dashboard and
    ``build_dataset`` refuses an event dataset outright, so neither could
    produce this number.
    """
    engine = trm._engine_for(_reference())
    scored = trm._full_test_accuracy(engine, "n_mnist")
    assert scored["test_samples"] == _TEST_SAMPLES
    assert 0.0 <= scored["test_accuracy"] <= 100.0


def test_event_scoring_reads_the_test_split_not_the_training_one(
    tonic_stub: None,
) -> None:
    """The held-out walk uses the test source, whose length differs."""
    engine = trm._engine_for(_reference())
    batches = list(trm._event_test_batches(engine))
    assert sum(len(labels) for _inputs, labels in batches) == _TEST_SAMPLES
    assert _TEST_SAMPLES != _TRAIN_SAMPLES


def test_the_progress_probe_works_without_an_image_loader(
    tonic_stub: None,
) -> None:
    """``build_loader`` refuses an event dataset; the probe cannot use it."""
    engine = trm._engine_for(_reference())
    trm._shrink_progress_evaluation(engine, "n_mnist")
    assert engine._test_batches is not None
    assert len(engine._test_batches) == 1
    cached = len(engine._test_batches[0][1])
    assert cached == min(trm.PROGRESS_SAMPLES, _TEST_SAMPLES)


# --- the shipped DVS128 Gesture row -------------------------------------


def test_the_gesture_row_takes_its_class_count_from_the_registry() -> None:
    """Restating `num_classes` would be a second place for it to drift."""
    from spikeforge.data.datasets import dataset_info

    row = next(
        r for r in trm.REFERENCES if r.dataset == "dvs128_gesture"
    )
    assert "num_classes" not in row.topology_params
    assert dataset_info("dvs128_gesture")[0] == 11
    assert row.topology_params["input_size"] == 128
    assert row.topology_params["in_channels"] == 2


def test_the_gesture_row_carries_verified_dataset_provenance() -> None:
    """DVS128 Gesture is CC BY 4.0, so the row publishes that, not a marker."""
    from spikeforge.data.dataset_provenance import dataset_provenance

    provenance = dataset_provenance("dvs128_gesture")
    assert provenance.license == "CC-BY-4.0"
    assert "Amir" in provenance.attribution


def test_every_reference_row_has_a_provenance_record() -> None:
    """A row whose dataset the table cannot describe is not publishable."""
    from spikeforge.data.dataset_provenance import dataset_provenance

    for row in trm.REFERENCES:
        assert dataset_provenance(row.dataset).attribution
