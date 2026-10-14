"""Event scoring and provenance for the reference-training script.

The published accuracy has to come from the *complete* held-out split
rather than the four batches the dashboard scores, and every shipped
reference row has to carry verified dataset provenance.
"""

from pathlib import Path

import pytest
from _event_helpers import (
    _TEST_SAMPLES,
    _TRAIN_SAMPLES,
    _FakeTonic,
    _reference,
)

import scripts.train_reference_models as trm
from spikeforge.data import event_loader
from spikeforge.events import event_source
from spikeforge.network import model_store


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
