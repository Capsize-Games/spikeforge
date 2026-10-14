"""Pick engines and run training, shrinking dashboard-scale work."""

import time
from typing import Any, Dict, List, Optional, Tuple

from train_reference_table import Reference

#: Samples in the small probe that replaces the engine's in-training
#: evaluation. See :func:`_shrink_progress_evaluation`.
PROGRESS_SAMPLES = 256


def _shrink_event_progress(engine: Any) -> None:
    """Shrink the progress probe on the event-bridged path.

    Bridging is per-sample, so the probe is sized in samples directly
    rather than by truncating a loader that does not exist here.
    """
    from spikeforge.training.event_batches import batch_event_samples

    source = engine._test_source
    count = min(PROGRESS_SAMPLES, source.size())
    engine._test_batches = [
        batch_event_samples(
            source, engine._spec, range(count), engine._bridge
        )
    ]


def _shrink_progress_evaluation(engine: Any, dataset: str) -> None:
    """Make the engine's in-training progress check cheap.

    ``EvalMixin`` scores four batches of 1000 held-out samples every fifth
    step -- ruinous for a full-dataset run (``conv_net`` spent over an hour
    in it) and unnecessary, since the published number comes from the full
    held-out pass in :func:`_full_test_accuracy`, not these checks. Shrunk to
    a single small batch; more so on the event path, where every probe
    sample is bridged rather than read from a tensor file.
    """
    if _is_event(dataset):
        _shrink_event_progress(engine)
        return
    from itertools import islice

    from spikeforge.data.data_loader import build_loader

    loader = build_loader(dataset, 1, PROGRESS_SAMPLES, train=False)
    engine._test_batches = list(islice(loader, 1))


def _is_event(dataset: str) -> bool:
    """Return True when a dataset is served through the event batch path."""
    from spikeforge.data.datasets import dataset_modality

    return dataset_modality(dataset) == "event"


def _engine_for(reference: Reference) -> Any:
    """Build the engine the dataset's modality selects.

    Branching on the registry's ``modality`` rather than on the dataset name
    means a new event dataset needs no change here.
    """
    if _is_event(reference.dataset):
        from spikeforge.training.event_engine import EventTrainingEngine

        return EventTrainingEngine(**reference.engine_kwargs())
    from spikeforge import TrainingEngine

    return TrainingEngine(**reference.engine_kwargs())


def _use_full_event_epoch(engine: Any, dataset: str) -> Optional[int]:
    """Make one event epoch a full pass over the training split.

    ``event_batches`` caps an epoch at ``EPOCH_BATCHES`` batches by default,
    which is right for the live dashboard and wrong for a published number: at
    ``subset=1`` it would still visit only ``10 * batch_size`` samples however
    large the split is, while the entry's notes claim the full training split.
    Naming the split's own size makes the claim true.

    Returns the epoch length, or ``None`` for an image dataset where the
    loader already walks the whole split.
    """
    if not _is_event(dataset):
        return None
    samples = engine._event_source.size()
    engine._epoch_samples = samples
    print(f"  event epoch: {samples} training samples (full split)")
    return samples


def _run_training(
    engine: Any, name: str
) -> Tuple[List[Dict[str, Any]], float]:
    """Run the training loop; return its per-epoch history and duration."""
    print(f"[{name}] training ...", flush=True)
    started = time.perf_counter()
    history: List[Dict[str, Any]] = []
    for metrics in engine.train():
        history.append(metrics)
    train_seconds = time.perf_counter() - started
    if not history:
        raise RuntimeError(f"{name}: training yielded no metrics")
    return history, train_seconds

