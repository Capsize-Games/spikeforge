"""Determinism and geometry of the synthetic event generators."""

import torch

from snn_interpreter.events.event_sample import EventSample
from snn_interpreter.events.synthetic import moving_dot, moving_edge

_FIELDS = ("x", "y", "t", "p")


def _equals(left: EventSample, right: EventSample) -> bool:
    """Return True when two samples agree on every coordinate tensor."""
    return all(
        torch.equal(getattr(left, field), getattr(right, field))
        for field in _FIELDS
    )


def test_moving_dot_is_deterministic() -> None:
    """The same seed yields byte-identical event streams."""
    first = moving_dot(num_steps=6, shape=(8, 8), seed=3)
    second = moving_dot(num_steps=6, shape=(8, 8), seed=3)
    assert _equals(first, second)


def test_moving_dot_positions_follow_velocity() -> None:
    """Positions are ``start + velocity * t`` rounded to the pixel grid."""
    dot = moving_dot(
        num_steps=3, shape=(10, 10), start=(1.0, 2.0), velocity=(2.0, 1.0)
    )
    assert dot.y.tolist() == [1, 3, 5]
    assert dot.x.tolist() == [2, 3, 4]


def test_moving_dot_clamps_to_sensor() -> None:
    """A fast dot is clamped inside the sensor bounds."""
    dot = moving_dot(
        num_steps=5, shape=(4, 4), start=(0.0, 0.0), velocity=(10.0, 10.0)
    )
    assert int(dot.y.max()) < 4
    assert int(dot.x.max()) < 4


def test_moving_dot_jitter_is_reproducible() -> None:
    """Jittered streams stay reproducible for a fixed seed."""
    left = moving_dot(num_steps=4, shape=(8, 8), jitter=2.0, seed=7)
    right = moving_dot(num_steps=4, shape=(8, 8), jitter=2.0, seed=7)
    assert _equals(left, right)


def test_moving_dot_jitter_perturbs_positions() -> None:
    """Jitter spreads the dot off its straight trajectory."""
    dot = moving_dot(num_steps=8, shape=(16, 16), jitter=2.0, seed=5)
    positions = dot.x.tolist() + dot.y.tolist()
    assert len(set(positions)) > 1


def test_moving_edge_is_deterministic_and_shaped() -> None:
    """The edge stream is reproducible and has one event per pixel-step."""
    first = moving_edge(num_steps=4, shape=(3, 5))
    second = moving_edge(num_steps=4, shape=(3, 5))
    assert _equals(first, second)
    assert first.num_events == 4 * 3


def test_moving_edge_shifts_by_speed() -> None:
    """The active column advances by ``speed`` per step."""
    edge = moving_edge(num_steps=3, shape=(1, 6), column=0, speed=2)
    assert edge.x.tolist() == [0, 2, 4]
