"""EventSample validation and its hand-computed dense accumulation."""

import pytest
import torch

from snn_interpreter.events.dense import to_frames, to_voxel
from snn_interpreter.events.event_sample import EventSample


def _sample() -> EventSample:
    """Return a tiny stream with one ON and one OFF event."""
    x = torch.tensor([0, 1, 3])
    y = torch.tensor([0, 2, 0])
    t = torch.tensor([0, 1, 2])
    p = torch.tensor([1, -1, 1])
    return EventSample(x, y, t, p, (3, 4), 3)


def test_fields_round_trip() -> None:
    """A valid stream exposes its layout and event count."""
    sample = _sample()
    assert sample.num_events == 3
    assert sample.shape == (3, 4)
    assert sample.num_steps == 3
    assert sample.p.tolist() == [1, -1, 1]


def test_to_frames_is_hand_computed() -> None:
    """Binary frames place ON in channel 0 and OFF in channel 1."""
    frames = to_frames(_sample())
    assert list(frames.shape) == [3, 2, 3, 4]
    assert float(frames[0, 0, 0, 0]) == 1.0
    assert float(frames[1, 1, 2, 1]) == 1.0
    assert float(frames[2, 0, 0, 3]) == 1.0
    assert float(frames.sum()) == 3.0
    assert set(torch.unique(frames).tolist()) <= {0.0, 1.0}


def test_to_voxel_counts_duplicates() -> None:
    """Voxel accumulation sums repeated events in the same bin."""
    x = torch.tensor([2, 2])
    y = torch.tensor([1, 1])
    t = torch.tensor([0, 0])
    p = torch.tensor([1, 1])
    voxel = to_voxel(EventSample(x, y, t, p, (2, 3), 1))
    assert float(voxel[0, 0, 1, 2]) == 2.0
    assert float(voxel.sum()) == 2.0


def test_empty_stream_is_valid_and_zero() -> None:
    """An empty stream is legal and accumulates to all zeros."""
    empty = torch.tensor([], dtype=torch.long)
    sample = EventSample(empty, empty, empty, empty, (2, 2), 1)
    assert sample.num_events == 0
    assert float(to_frames(sample).sum()) == 0.0


def test_mismatched_lengths_are_rejected() -> None:
    """Unequal x/y/t/p lengths raise a ``ValueError``."""
    with pytest.raises(ValueError):
        EventSample(
            torch.tensor([0, 1]),
            torch.tensor([0]),
            torch.tensor([0, 1]),
            torch.tensor([1, 1]),
            (2, 2),
            2,
        )


@pytest.mark.parametrize(
    "bad_x,bad_y,bad_t",
    [(2, 0, 0), (0, 2, 0), (0, 0, 2)],
)
def test_out_of_range_indices_are_rejected(
    bad_x: int, bad_y: int, bad_t: int
) -> None:
    """An out-of-range x, y, or t index raises a ``ValueError``."""
    with pytest.raises(ValueError):
        EventSample(
            torch.tensor([bad_x]),
            torch.tensor([bad_y]),
            torch.tensor([bad_t]),
            torch.tensor([1]),
            (2, 2),
            2,
        )


def test_bad_polarity_is_rejected() -> None:
    """A polarity other than ``+1``/``-1`` raises a ``ValueError``."""
    with pytest.raises(ValueError):
        EventSample(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            (2, 2),
            2,
        )


def test_non_positive_layout_is_rejected() -> None:
    """A zero-sized sensor or step count raises a ``ValueError``."""
    with pytest.raises(ValueError):
        EventSample(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),
            (0, 2),
            2,
        )
