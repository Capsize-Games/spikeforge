"""Public exports for the spikeforge package.

:class:`TrainingEngine` is the entry point for new code: it owns the
cancellable training loop, topology selection, encoding, checkpointing, and
evaluation, and yields a metrics dict per epoch. :class:`SNNTrainer` is the
older, narrower helper that loads an MNIST subset and rate-codes it -- it
backs the tutorial demo and the animation walkthroughs, and it is not the
class to reach for when training a network.

The ML exports are resolved on first access (PEP 562) rather than at import
time, so ``import spikeforge`` no longer pulls ``torch`` / ``snntorch`` /
``torchvision``. That is what lets the non-ML surfaces —
``spikeforge.observability`` and ``spikeforge.config`` — be imported and
tested without the ML stack. ``from spikeforge import TrainingEngine`` and the
attribute names below are unchanged.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from spikeforge.version import __version__

if TYPE_CHECKING:
    from spikeforge.encoding.delta_trainer import DeltaTrainer
    from spikeforge.encoding.latency_trainer import (
        LatencyTrainer,
        convert_to_time,
    )
    from spikeforge.encoding.random_spikegen import RandomSpikeGenerator
    from spikeforge.training.logger import SNNTrainerLogger
    from spikeforge.training.trainer import SNNTrainer
    from spikeforge.training.training_engine import TrainingEngine

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

#: Public export -> defining module, resolved on first attribute access.
_EXPORTS = {
    "DeltaTrainer": "spikeforge.encoding.delta_trainer",
    "LatencyTrainer": "spikeforge.encoding.latency_trainer",
    "convert_to_time": "spikeforge.encoding.latency_trainer",
    "RandomSpikeGenerator": "spikeforge.encoding.random_spikegen",
    "SNNTrainerLogger": "spikeforge.training.logger",
    "SNNTrainer": "spikeforge.training.trainer",
    "TrainingEngine": "spikeforge.training.training_engine",
}


def __getattr__(name: str) -> Any:
    """Resolve a public ML export on first access (PEP 562)."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
