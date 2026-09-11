"""Event sample source: tonic selection and explicit synthetic fallback."""

import json
from typing import Tuple

import pytest

from spikeforge.events import event_source
from spikeforge.events.event_sample import EventSample
from spikeforge.events.synthetic import moving_dot


def _fake_pair(
    name: str, index: int, num_steps: int, save_to: object
) -> Tuple[EventSample, int]:
    """Return a tiny deterministic pair standing in for tonic."""
    sample = moving_dot(num_steps=num_steps, shape=(8, 8), seed=index)
    return sample, index % 10


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
    monkeypatch.setattr(event_source, "dataset_available", lambda name: True)
    monkeypatch.setattr(
        event_source.event_loader, "load_event_pair", _fake_pair
    )
    src = event_source.EventSampleSource("n_mnist")
    assert src.origin == event_source.TONIC
    assert "tonic" in src.description
    sample, label = src.load(2)
    assert isinstance(sample, EventSample)
    assert label == 2


def test_missing_loader_falls_back_to_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the optional extra the source uses the explicit offline path."""
    monkeypatch.setattr(event_source, "dataset_available", lambda name: False)
    src = event_source.EventSampleSource("dvs128_gesture")
    assert src.origin == event_source.SYNTHETIC
    assert "offline" in src.description
