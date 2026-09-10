"""Configuration for a topology benchmark run.

The benchmark lives in its own package, not under ``introspection``: it
measures wall time and memory of an execution run rather than describing a
recorded trajectory, and it exposes its own tiny, JSON-returning fixture.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BenchmarkConfig:
    """A small, explicit benchmark fixture.

    ``topologies`` are registry topology names. ``steps`` and ``batch_size``
    shape the random spike input, ``seed`` fixes both the module weights and
    the spikes, and ``warmup`` untimed runs precede ``repeats`` timed samples.
    ``compiled`` additionally measures a ``torch.compile`` production path and
    ``backward`` measures a backward pass where one is meaningful.
    """

    topologies: Tuple[str, ...] = ("fc_small",)
    batch_size: int = 2
    steps: int = 4
    repeats: int = 2
    warmup: int = 1
    seed: int = 0
    device: str = "auto"
    compiled: bool = False
    backward: bool = True


def default_config() -> BenchmarkConfig:
    """Return the tiny default fixture so tests and CI stay fast."""
    return BenchmarkConfig()
