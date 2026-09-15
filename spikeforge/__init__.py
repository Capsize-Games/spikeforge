"""Public exports for the spikeforge package.

:class:`TrainingEngine` is the entry point for new code: it owns the
cancellable training loop, topology selection, encoding, checkpointing, and
evaluation, and yields a metrics dict per epoch. :class:`SNNTrainer` is the
older, narrower helper that loads an MNIST subset and rate-codes it -- it
backs the tutorial demo and the animation walkthroughs, and it is not the
class to reach for when training a network.
"""

from spikeforge.encoding.delta_trainer import DeltaTrainer
from spikeforge.encoding.latency_trainer import (
    LatencyTrainer,
    convert_to_time,
)
from spikeforge.encoding.random_spikegen import RandomSpikeGenerator
from spikeforge.training.logger import SNNTrainerLogger
from spikeforge.training.trainer import SNNTrainer
from spikeforge.training.training_engine import TrainingEngine
from spikeforge.version import __version__

__all__ = [
    "__version__",
    "TrainingEngine",
    "SNNTrainer",
    "SNNTrainerLogger",
    "LatencyTrainer",
    "DeltaTrainer",
    "RandomSpikeGenerator",
    "convert_to_time",
]
