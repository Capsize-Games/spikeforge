"""Event sample source: tonic selection and explicit synthetic fallback."""

import json
from typing import Tuple

import pytest

from spikeforge.events import event_source
from spikeforge.events.event_sample import EventSample
from spikeforge.events.synthetic import moving_dot


class _FakeDataset:
    """A tonic-shaped dataset that counts how often it is constructed."""

    sensor_size = (8, 8, 2)
    #: Constructions since the last reset, so the open-once rule is testable.
    opens = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Count this construction."""
        type(self).opens += 1

    def __len__(self) -> int:
        """Return a split length distinguishable from the synthetic window."""
        return 7


def _fake_pair(
    dataset: object, index: int, num_steps: int = 10
) -> Tuple[EventSample, int]:
    """Return a tiny deterministic pair standing in for tonic."""
    sample = moving_dot(num_steps=num_steps, shape=(8, 8), seed=index)
    return sample, index % 10


def _use_fake_tonic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the source's tonic path at the real seams, offline."""
    monkeypatch.setattr(event_source, "dataset_available", lambda name: True)
    monkeypatch.setattr(
        event_source.event_loader.tonic_api, "dataset_class",
        lambda name: _FakeDataset,
    )
    monkeypatch.setattr(
        event_source.event_loader, "sample_from", _fake_pair
    )
    _FakeDataset.opens = 0


def test_synthetic_source_is_explicit() -> None:
    """Forcing synthetic marks the origin and description honestly."""
    src = event_source.EventSampleSource("n_mnist", synthetic_only=True)
    assert src.origin == event_source.SYNTHETIC
    assert "synthetic" in src.description
    assert src.info()["modality"] == "event"
    assert json.dumps(src.info())


def test_synthetic_clamp_wraps_and_load_repeats() -> None:
    """Synthetic indices wrap and repeat exactly for a given index."""
    src = event_source.EventSampleSource("n_mnist", synthetic_only=True)
    assert src.clamp(event_source.SYNTHETIC_SIZE + 3) == 3
    first, label = src.load(3)
    again, label2 = src.load(3)
    assert label == label2
    assert first.x.tolist() == again.x.tolist()


def test_synthetic_stream_has_both_polarities() -> None:
    """The offline stream carries ON and OFF events on the set sensor."""
    src = event_source.EventSampleSource("n_mnist", synthetic_only=True)
    sample, _ = src.load(0)
    assert set(sample.p.tolist()) == {1, -1}
    assert sample.shape == event_source.SYNTHETIC_SHAPE


def test_tonic_backend_selected_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An available loader routes through the tonic path."""
    _use_fake_tonic(monkeypatch)
    src = event_source.EventSampleSource("n_mnist")
    assert src.origin == event_source.TONIC
    assert "tonic" in src.description
    sample, label = src.load(2)
    assert isinstance(sample, EventSample)
    assert label == 2


def test_the_tonic_dataset_is_opened_once_not_once_per_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading an epoch must not pay the split's indexing cost per sample.

    Opening a tonic dataset indexes the split's files. The source used to
    open one inside every ``load``, so a 1000-sample epoch indexed the split
    1000 times -- which would have dominated any event training run and any
    timing measured from one.
    """
    _use_fake_tonic(monkeypatch)
    src = event_source.EventSampleSource("n_mnist")
    assert _FakeDataset.opens == 0, "constructing a source must not open it"
    for index in range(25):
        src.load(index)
    assert _FakeDataset.opens == 1


def test_a_tonic_source_reports_the_real_split_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``size`` is the split's own length, not the synthetic window."""
    _use_fake_tonic(monkeypatch)
    assert event_source.EventSampleSource("n_mnist").size() == 7


def test_a_synthetic_source_reports_its_generated_window() -> None:
    """Offline, ``size`` is the width of the window this split generates."""
    train = event_source.EventSampleSource("n_mnist", synthetic_only=True)
    test = event_source.EventSampleSource(
        "n_mnist", synthetic_only=True, split="test"
    )
    assert train.size() == event_source.SYNTHETIC_SIZE
    assert test.size() == event_source.SYNTHETIC_TEST_SIZE


def test_missing_loader_falls_back_to_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the optional extra the source uses the explicit offline path."""
    monkeypatch.setattr(event_source, "dataset_available", lambda name: False)
    src = event_source.EventSampleSource("dvs128_gesture")
    assert src.origin == event_source.SYNTHETIC
    assert "offline" in src.description
