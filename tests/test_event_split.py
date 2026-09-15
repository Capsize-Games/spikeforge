"""The event train/test split: real, per-dataset, and never silent.

Before this split existed, ``EventTrainingEngine._epoch_batches`` accepted a
``train`` flag and ignored it, so every event "test accuracy" was measured on
the head of the training stream. These tests fail if that regresses: they
assert the two splits return *different data*, not merely that a flag is
accepted, and that a dataset shipping no held-out partition says so by name
instead of handing back training data.
"""

from typing import Any, Dict, List, Tuple

import pytest
import torch

from spikeforge.data import datasets, event_loader
from spikeforge.data.event_errors import EventSplitMissingError
from spikeforge.events import event_source
from spikeforge.events.event_source import EventSampleSource
from spikeforge.network import model_store
from spikeforge.training.eval_mixin import EVAL_BATCHES
from spikeforge.training.event_engine import EventTrainingEngine

_STEPS = 4


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint writes into a per-test temporary directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _engine(**kwargs: Any) -> EventTrainingEngine:
    """Return an offline CPU engine with small, download-free defaults."""
    defaults: Dict[str, Any] = {
        "dataset": "n_mnist", "synthetic_only": True, "num_steps": _STEPS,
        "subset": 10, "batch_size": 4, "hidden": 8, "epochs": 1,
        "device": "cpu",
    }
    defaults.update(kwargs)
    return EventTrainingEngine(**defaults)


def _source(split: str) -> EventSampleSource:
    """Return an offline source for one split of the default dataset."""
    return EventSampleSource(
        "n_mnist", synthetic_only=True, num_steps=_STEPS, split=split
    )


def _indices(source: EventSampleSource) -> List[int]:
    """Return every distinct index the source's clamp can produce."""
    return sorted({source.clamp(i) for i in range(500)})


class _RecordingDataset:
    """A tonic-shaped dataset whose samples encode its own split kwargs."""

    sensor_size = (8, 8, 2)
    #: Every instance's constructor kwargs, in construction order.
    seen: List[Dict[str, Any]] = []

    def __init__(self, root: str, **kwargs: Any) -> None:
        """Record the kwargs the loader resolved for this split."""
        self._kwargs = kwargs
        type(self).seen.append(dict(kwargs))
        #: A test split of a different length, as tonic's really are.
        self._size = 3 if kwargs.get("train") else 2

    def __len__(self) -> int:
        """Return this split's sample count."""
        return self._size

    def __getitem__(self, index: int) -> Tuple[Dict[str, List[int]], int]:
        """Return a stream whose column offset differs between splits."""
        column = 0 if self._kwargs.get("train") else 4
        stream = {
            "x": [column, column + 1],
            "y": [0, 1],
            "t": [0, 1000],
            "p": [1, 0],
        }
        return stream, index % 2


# --- the registry declares each split explicitly ------------------------


def test_registry_declares_each_dataset_own_split_arguments() -> None:
    """Split selection is per dataset, not one boolean for all of them."""
    assert datasets.dataset_split_kwargs("n_mnist", "train") == {"train": True}
    assert datasets.dataset_split_kwargs("n_mnist", "test") == {"train": False}
    assert datasets.dataset_split_kwargs("dvs128_gesture", "test") == {
        "train": False
    }
    assert datasets.dataset_split_kwargs("ssc", "train") == {"split": "train"}
    assert datasets.dataset_split_kwargs("ssc", "test") == {"split": "test"}


def test_dataset_without_a_test_split_is_named_not_substituted() -> None:
    """CIFAR10-DVS ships one pool, so asking for held-out data raises."""
    assert datasets.dataset_splits("cifar10_dvs") == ["train"]
    with pytest.raises(EventSplitMissingError) as ctx:
        datasets.dataset_split_kwargs("cifar10_dvs", "test")
    message = str(ctx.value)
    assert "cifar10_dvs" in message
    assert "test" in message
    assert "held-out" in message


def test_source_rejects_an_undeclared_split_on_either_backend() -> None:
    """The check is a registry fact, so it fires without tonic installed."""
    with pytest.raises(EventSplitMissingError):
        EventSampleSource("cifar10_dvs", synthetic_only=True, split="test")


def test_engine_refuses_a_dataset_it_cannot_score_honestly() -> None:
    """No test split means no training run that could report one."""
    with pytest.raises(EventSplitMissingError) as ctx:
        _engine(dataset="cifar10_dvs")
    assert "cifar10_dvs" in str(ctx.value)


# --- the engine reads two different streams -----------------------------


def test_epoch_batches_differ_between_the_two_splits() -> None:
    """The held-out batches are different data, not the training head."""
    engine = _engine()
    train_inputs, train_labels = engine._epoch_batches(train=True)[0]
    test_inputs, test_labels = engine._epoch_batches(train=False)[0]
    assert train_inputs.shape == test_inputs.shape
    assert not torch.equal(train_inputs, test_inputs)
    assert not torch.equal(train_labels, test_labels)


def test_load_test_batches_reads_the_held_out_split() -> None:
    """Cached evaluation batches come from the test source, truncated."""
    engine = _engine()
    cached = engine._load_test_batches()
    expected = engine._epoch_batches(train=False)[:EVAL_BATCHES]
    assert len(cached) == len(expected)
    for (inputs, labels), (want_in, want_lab) in zip(cached, expected):
        assert torch.equal(inputs, want_in)
        assert torch.equal(labels, want_lab)
    train_inputs, _ = engine._epoch_batches(train=True)[0]
    assert not torch.equal(cached[0][0], train_inputs)


def test_engine_holds_one_source_per_split() -> None:
    """Each source knows its own split rather than taking a flag."""
    engine = _engine()
    assert engine._event_source.split == "train"
    assert engine._test_source.split == "test"


def test_checkpoint_records_the_held_out_stream_provenance() -> None:
    """The card names the stream the accuracy was measured on."""
    torch.manual_seed(0)
    _engine().save("event_split_fixture")
    meta = model_store.load("event_split_fixture")["meta"]
    assert meta["event_test_origin"] == "synthetic"
    assert "held-out" in meta["event_test_description"]
    assert "no real recording" in meta["event_test_description"]


# --- the synthetic backend holds a genuinely disjoint pool ---------------


def test_synthetic_splits_draw_from_disjoint_index_ranges() -> None:
    """A generated held-out sample is never one the training stream served."""
    train, test = _indices(_source("train")), _indices(_source("test"))
    assert train == list(range(event_source.SYNTHETIC_SIZE))
    assert test == list(range(
        event_source.SYNTHETIC_SIZE,
        event_source.SYNTHETIC_SIZE + event_source.SYNTHETIC_TEST_SIZE,
    ))
    assert not set(train) & set(test)


def test_synthetic_split_samples_never_coincide() -> None:
    """No generated test sample equals any sample the training pool holds."""
    test_source = _source("test")
    train_source = _source("train")
    train_columns = {
        tuple(train_source.load(index)[0].x.tolist())
        for index in _indices(train_source)
    }
    for index in _indices(test_source):
        sample, _label = test_source.load(index)
        assert tuple(sample.x.tolist()) not in train_columns


def test_synthetic_training_stream_is_unchanged_by_the_split() -> None:
    """Adding the split left the training pool's samples exactly as they were.

    The offline generator is a fixture other tests and the example script
    compare against, so the default split must still produce the stream it
    always did.
    """
    source = _source("train")
    assert source.clamp(event_source.SYNTHETIC_SIZE + 3) == 3
    sample, label = source.load(3)
    assert sample.x.tolist() == [5, 6, 7, 8, 17, 16, 15, 14]
    assert label == 3
    assert "no real recording" in source.description
    assert "held-out" not in source.description


def test_synthetic_split_provenance_says_it_is_generated() -> None:
    """The held-out description still refuses to claim a recording."""
    info = _source("test").info()
    assert info["split"] == "test"
    assert info["origin"] == event_source.SYNTHETIC
    assert "held-out" in info["description"]
    assert "no real recording" in info["description"]


# --- under tonic, each split opens a different dataset -------------------


def test_tonic_splits_open_different_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each split reaches tonic with its own kwargs and its own content."""
    _RecordingDataset.seen = []
    monkeypatch.setattr(
        event_loader.tonic_api, "dataset_class",
        lambda name: _RecordingDataset,
    )
    monkeypatch.setattr(event_source, "dataset_available", lambda name: True)
    train = EventSampleSource("n_mnist", num_steps=_STEPS, split="train")
    test = EventSampleSource("n_mnist", num_steps=_STEPS, split="test")
    train_sample, _ = train.load(0)
    test_sample, _ = test.load(0)
    assert _RecordingDataset.seen == [{"train": True}, {"train": False}]
    assert train_sample.x.tolist() != test_sample.x.tolist()


def test_tonic_split_lengths_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two splits are separate datasets, so their sizes differ."""
    monkeypatch.setattr(
        event_loader.tonic_api, "dataset_class",
        lambda name: _RecordingDataset,
    )
    train = event_loader._dataset("n_mnist", None, "train")
    test = event_loader._dataset("n_mnist", None, "test")
    assert len(train) != len(test)
