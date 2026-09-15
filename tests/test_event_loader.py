"""Event loader conversion contract and its typed absence error."""

import sys
from typing import Any, Dict, List

import numpy as np
import pytest

from spikeforge.data import event_loader
from spikeforge.data.event_errors import (
    EventsExtraMissingError,
    EventTimestampError,
)


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


def _cochlea_stream() -> Any:
    """Return a tonic SSC/SHD-shaped ``(t, x, p)`` array with no ``y``."""
    dtype = np.dtype([("t", int), ("x", int), ("p", int)])
    return np.array(
        [(0, 5, 1), (500_000, 699, 1), (1_000_000, 0, 1)], dtype=dtype
    )


def test_a_one_row_sensor_needs_no_y_field() -> None:
    """An auditory sensor has channels, not pixel rows.

    Tonic's SHD and SSC declare ``sensor_size = (700, 1, 1)`` and carry no
    ``y`` at all, so every event belongs on row 0 rather than failing for an
    axis the recording does not have.
    """
    sample = event_loader.events_to_sample(_cochlea_stream(), (1, 700), 4)
    assert sample.num_events == 3
    assert sample.shape == (1, 700)
    assert sample.x.tolist() == [5, 699, 0]
    assert sample.y.tolist() == [0, 0, 0]
    assert sample.p.tolist() == [1, 1, 1]


def test_a_taller_sensor_still_requires_y() -> None:
    """Defaulting ``y`` on a 2-D sensor would flatten a real recording."""
    with pytest.raises(ValueError) as ctx:
        event_loader.events_to_sample(_cochlea_stream(), (128, 128), 4)
    assert "'y'" in str(ctx.value)


def test_a_numpy_stream_gets_the_named_error_too() -> None:
    """A structured array raises ValueError, not KeyError, for a bad field.

    The named message has to fire for both container shapes, or a real tonic
    array reports numpy's wording instead of this project's.
    """
    dtype = np.dtype([("t", int), ("y", int), ("p", int)])
    stream = np.array([(0, 1, 1)], dtype=dtype)
    with pytest.raises(ValueError) as ctx:
        event_loader.events_to_sample(stream, (1, 700), 4)
    assert "event stream has no 'x' field" in str(ctx.value)


def test_impossible_timestamps_are_refused_not_binned() -> None:
    """Destroyed timing must fail by name, not train quietly.

    Tonic's SHD/SSC reader scales float16 seconds by 1e6; the multiply
    overflows to inf, becomes NaN, and every timestamp casts to INT64_MIN.
    Binning would then see a zero-width span and put every event in the first
    time step -- a number measured on that is worthless and looks fine.
    """
    dtype = np.dtype([("t", int), ("x", int), ("p", int)])
    broken = np.array(
        [(np.iinfo(np.int64).min, 5, 1), (np.iinfo(np.int64).min, 9, 1)],
        dtype=dtype,
    )
    with pytest.raises(EventTimestampError) as ctx:
        event_loader.events_to_sample(broken, (1, 700), 25)
    message = str(ctx.value)
    assert "float16" in message
    assert "one time bin" in message


def test_a_genuinely_instantaneous_sample_is_still_accepted() -> None:
    """Every event sharing one timestamp is legitimate, not corruption.

    The guard keys on impossible values rather than on a zero-width span,
    because a zero-width span is something a real recording can have.
    """
    same: Dict[str, List[int]] = {
        "x": [1, 2], "y": [0, 0], "t": [7, 7], "p": [1, 1],
    }
    sample = event_loader.events_to_sample(same, (4, 4), 3)
    assert sample.t.tolist() == [0, 0]


def test_zero_timestamps_are_accepted() -> None:
    """Zero is a valid recording time; only negatives are impossible."""
    zeros: Dict[str, List[int]] = {
        "x": [0], "y": [0], "t": [0], "p": [1],
    }
    assert event_loader.events_to_sample(zeros, (4, 4), 2).num_events == 1


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
