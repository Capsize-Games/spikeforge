"""Score references and write their records and checkpoints."""

import hashlib
import platform
import time
from pathlib import Path
from typing import Any, Dict, List

from train_reference_engine import (
    _engine_for,
    _is_event,
    _run_training,
    _shrink_progress_evaluation,
    _use_full_event_epoch,
)
from train_reference_table import REPO_ROOT, Reference

#: Batch size used for scoring only; larger than the training batch because
#: no gradients are held.
EVAL_BATCH = 512


def _digest(path: Path) -> str:
    """Return the SHA-256 of a checkpoint, for the catalog to pin."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hardware() -> Dict[str, Any]:
    """Describe the machine well enough for a reader to calibrate timings."""
    import torch

    return {
        "device": "cpu",
        "processor": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def _score_batches(engine: Any, batches: Any) -> Dict[str, Any]:
    """Score an iterable of ``(inputs, targets)`` batches end to end."""
    import torch

    correct = total = 0
    started = time.perf_counter()
    with torch.no_grad():
        for inputs, targets in batches:
            predicted = engine.predict(inputs).cpu()
            correct += int((predicted == targets).sum())
            total += int(len(targets))
    return {
        "test_accuracy": round(100.0 * correct / max(total, 1), 2),
        "test_samples": total,
        "eval_seconds": round(time.perf_counter() - started, 1),
    }


def _event_test_batches(engine: Any) -> Any:
    """Yield the engine's **whole** held-out event split, in eval batches.

    ``build_dataset`` refuses an event dataset by design, and the engine's own
    ``_load_test_batches`` truncates to the four batches the dashboard scores,
    so neither can produce a published number. This walks the test source's
    full length instead -- every sample, bridged through the same path
    training used -- so an event row means what the image rows mean.
    """
    from spikeforge.training.event_batches import batch_event_samples

    source = engine._test_source
    total = source.size()
    for start in range(0, total, EVAL_BATCH):
        stop = min(start + EVAL_BATCH, total)
        yield batch_event_samples(
            source, engine._spec, range(start, stop), engine._bridge
        )


def _full_test_accuracy(engine: Any, dataset: str) -> Dict[str, Any]:
    """Score the complete held-out split, not a sample of it.

    ``TrainingEngine.evaluate`` deliberately scores four cached batches so the
    live dashboard stays responsive. A published number has to be the whole
    test set, so this walks it end to end -- identically for both modalities,
    because a row a reader compares against another row has to mean the same
    thing in both.
    """
    from spikeforge.data.data_loader import build_loader

    if _is_event(dataset):
        return _score_batches(engine, _event_test_batches(engine))
    return _score_batches(
        engine, build_loader(dataset, 1, EVAL_BATCH, train=False)
    )


def _checkpoint_payload(
    engine: Any, history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Return the lean checkpoint payload: weights, meta, manifest."""
    return {
        "state_dict": engine.net.state_dict(),
        "meta": engine._meta(),
        "history": history[-1:],
        "manifest": engine._manifest(history[-1:]),
        "saved_at": time.time(),
    }


def _save_checkpoint(
    engine: Any,
    reference: Reference,
    history: List[Dict[str, Any]],
    out: Path,
) -> Path:
    """Write a lean checkpoint: weights, metadata card, and manifest.

    ``TrainingEngine.save`` writes into the shared model store and keeps the
    full per-step metric history, which more than doubles the file. These
    artifacts ship inside the ``spikeforge-hub`` wheel, so the history is
    dropped and only the last metrics are kept -- the reproducibility manifest
    still records the seed and the whole configuration, which is what makes
    the run repeatable.
    """
    import torch

    path = out / f"{reference.name}.pt"
    torch.save(_checkpoint_payload(engine, history), path)
    return path


def _reference_fields(reference: Reference) -> Dict[str, Any]:
    """Return the reference-config fields of a result record."""
    return {
        "name": reference.name,
        "dataset": reference.dataset,
        "topology": reference.topology,
        "epochs": reference.epochs,
        "num_steps": reference.num_steps,
        "hidden": reference.hidden,
        "beta": reference.beta,
        "batch_size": reference.batch_size,
        "subset": reference.subset,
        "lr": reference.lr,
        "seed": reference.seed,
        "note": reference.note,
        "command": reference.command(),
    }


def _build_record(
    reference: Reference,
    history: List[Dict[str, Any]],
    train_seconds: float,
    scored: Dict[str, Any],
    checkpoint: Path,
) -> Dict[str, Any]:
    """Assemble the JSON-ready result record for one trained config."""
    last = history[-1]
    record = _reference_fields(reference)
    record.update(
        final_loss=round(float(last["loss"]), 4),
        train_seconds=round(train_seconds, 1),
        checkpoint=str(checkpoint.relative_to(REPO_ROOT)),
        checkpoint_bytes=checkpoint.stat().st_size,
        sha256=_digest(checkpoint),
        **scored,
    )
    return record


def _print_train_summary(name: str, record: Dict[str, Any]) -> None:
    """Print one line summarizing a trained config's held-out score."""
    print(
        f"[{name}] {record['test_accuracy']}% on "
        f"{record['test_samples']} held-out samples, "
        f"{record['train_seconds']}s",
        flush=True,
    )


def _train(reference: Reference, out: Path) -> Dict[str, Any]:
    """Train one configuration, score it, and persist its checkpoint."""
    engine = _engine_for(reference)
    _use_full_event_epoch(engine, reference.dataset)
    _shrink_progress_evaluation(engine, reference.dataset)
    history, train_seconds = _run_training(engine, reference.name)
    scored = _full_test_accuracy(engine, reference.dataset)
    checkpoint = _save_checkpoint(engine, reference, history, out)
    record = _build_record(
        reference, history, train_seconds, scored, checkpoint
    )
    _print_train_summary(reference.name, record)
    return record

