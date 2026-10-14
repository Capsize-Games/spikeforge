"""A serving fixture over one ``.spkf`` bundle."""

from dataclasses import dataclass
from typing import Tuple

#: Mode label the serving records carry; the gate matches on it.
MODE = "serving"
#: Default raw sample geometry when the bundle pins no ``input_size``.
SAMPLE_SIZE: Tuple[int, int] = (28, 28)


@dataclass(frozen=True)
class ServingBenchmarkConfig:
    """A serving fixture over one ``.spkf`` bundle.

    ``calls`` predict calls are issued with ``concurrency`` worker sessions,
    each call advancing the session through one encoded sample. ``warmup``
    untimed calls precede the timed ones, and ``seed`` fixes the raw sample.
    """

    bundle: str
    calls: int = 8
    concurrency: int = 1
    warmup: int = 1
    seed: int = 0
    device: str = "cpu"
