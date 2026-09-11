"""Event loader conversion contract and its typed absence error."""

import sys
from typing import Dict, List

import pytest

from spikeforge.data import event_loader
from spikeforge.data.event_errors import EventsExtraMissingError


def _stream() -> Dict[str, List[int]]:
    """Return a minimal tonic-shaped ``(x, y, t, p)`` mapping."""
    return {
        "x": [0, 1, 2, 3],
        "y": [0, 1, 2, 3],
        "t": [0, 1000, 2000, 3000],
        "p": [1, 0, 1, 0],
    }


def test_conversion_maps_coordinates_and_bins_time() -> None:
    """The pure conversion produces the documented EventSample fields."""
    sample = event_loader.events_to_sample(_stream(), (8, 8), num_steps=4)
    assert sample.num_events == 4
    assert sample.shape == (8, 8)
    assert sample.num_steps == 4
    assert sample.x.tolist() == [0, 1, 2, 3]
    assert sample.y.tolist() == [0, 1, 2, 3]
    assert sample.t.tolist() == [0, 1, 2, 3]
    assert sample.p.tolist() == [1, -1, 1, -1]


def test_conversion_accepts_empty_stream() -> None:
    """An empty stream yields an empty but valid sample."""
    empty: Dict[str, List[int]] = {"x": [], "y": [], "t": [], "p": []}
    sample = event_loader.events_to_sample(empty, (4, 4), num_steps=3)
    assert sample.num_events == 0


def test_conversion_rejects_missing_field() -> None:
    """A stream without a coordinate field fails clearly."""
    partial: Dict[str, List[int]] = {"x": [0], "y": [0], "t": [0]}
    with pytest.raises(ValueError):
        event_loader.events_to_sample(partial, (4, 4), 2)


def test_loader_names_the_events_extra_when_tonic_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing tonic raises a typed error naming the `events` extra."""
    monkeypatch.setitem(sys.modules, "tonic", None)
    monkeypatch.setitem(sys.modules, "tonic.datasets", None)
    with pytest.raises(EventsExtraMissingError) as ctx:
        event_loader.load_event_sample("n_mnist", num_steps=4)
    message = str(ctx.value)
    assert "events" in message
    assert "tonic" in message
    assert "n_mnist" in message


def test_ensure_fails_the_same_way_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The download entry point raises the same typed error when absent."""
    monkeypatch.setitem(sys.modules, "tonic", None)
    monkeypatch.setitem(sys.modules, "tonic.datasets", None)
    with pytest.raises(EventsExtraMissingError):
        event_loader.ensure_event_dataset("ssc")


def test_loader_rejects_image_dataset() -> None:
    """Asking the event loader for an image dataset fails clearly."""
    with pytest.raises(ValueError):
        event_loader.load_event_sample("mnist")
