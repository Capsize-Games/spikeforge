"""Deterministic synthetic event generators for offline tests and demos.

Both generators fabricate an :class:`EventSample` without any network
access so tests and dashboards can exercise the event pipeline offline. A
seeded generator never touches the global RNG, so ``seed`` alone fixes the
stream and repeated calls return identical events.
"""

from typing import Optional, Tuple

import torch

from spikeforge.events.event_sample import EventSample

Shape = Tuple[int, int]


def _positions(
    start: Tuple[float, float],
    velocity: Tuple[float, float],
    times: torch.Tensor,
    jitter: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the rounded ``(row, column)`` trajectory of a moving dot."""
    rows = start[0] + velocity[0] * times.float()
    cols = start[1] + velocity[1] * times.float()
    if jitter:
        generator = torch.Generator().manual_seed(int(seed))
        noise = torch.rand(2, times.numel(), generator=generator) - 0.5
        rows = rows + jitter * noise[0]
        cols = cols + jitter * noise[1]
    return rows.round(), cols.round()


def moving_dot(
    num_steps: int = 10,
    shape: Shape = (28, 28),
    start: Tuple[float, float] = (4.0, 4.0),
    velocity: Tuple[float, float] = (1.0, 1.0),
    polarity: int = 1,
    jitter: float = 0.0,
    seed: int = 0,
) -> EventSample:
    """Return one event per step for a dot moving at constant velocity.

    Positions are ``start + velocity * t`` rounded to the nearest pixel and
    clamped to the sensor. ``jitter`` perturbs each position by up to half a
    pixel using a local seeded generator, so the stream is reproducible for
    a given ``seed`` whether or not jitter is enabled.
    """
    height, width = shape
    times = torch.arange(num_steps)
    rows, cols = _positions(start, velocity, times, jitter, seed)
    rows = rows.clamp(0, height - 1).long()
    cols = cols.clamp(0, width - 1).long()
    values = torch.full((num_steps,), int(polarity))
    return EventSample(cols, rows, times, values, shape, num_steps)


def moving_edge(
    num_steps: int = 10,
    shape: Shape = (28, 28),
    column: Optional[int] = None,
    speed: int = 1,
    polarity: int = 1,
) -> EventSample:
    """Return a vertical edge that shifts by ``speed`` columns per step.

    Every row of the active column fires at each step, so the stream has
    ``num_steps * H`` events and is fully deterministic.
    """
    height, width = shape
    start = width // 2 if column is None else int(column)
    times = torch.arange(num_steps)
    columns = (start + speed * times) % width
    rows = torch.arange(height)
    y = rows.repeat(num_steps)
    x = columns.repeat_interleave(height)
    t = times.repeat_interleave(height)
    p = torch.full((y.numel(),), int(polarity))
    return EventSample(x, y, t, p, shape, num_steps)
