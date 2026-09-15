#!/usr/bin/env python3
"""Train the published reference configurations and record what they score.

The project could not answer "what does this actually achieve on MNIST, in
how long, on what hardware?" with a link. The machinery to answer it was all
shipped -- training, evaluation, checkpointing, the reproducibility manifest
-- so this script is the missing piece: it runs a fixed table of reference
configurations, scores each on the **complete** held-out test set (not the
four cached batches the live dashboard scores against), and writes both a
machine-readable result set and the Markdown table that goes in
``documentation/benchmarks.md``.

These are deliberately *reference* configurations, not tuned attempts at
state of the art: modest epoch counts, stock hyperparameters, one seed, CPU.
The point is a number a reader can reproduce with the command printed beside
it, and a checkpoint the model hub can carry.

Usage::

    python scripts/train_reference_models.py                 # all of them
    python scripts/train_reference_models.py --only mnist-fc-legacy
    python scripts/train_reference_models.py --list
    python scripts/train_reference_models.py --sync-provenance

Outputs (under ``--out``, default ``build/reference/``):

* ``results.json``  -- one record per configuration, with hardware metadata
* ``results.md``    -- the Markdown table for the documentation page
* ``<name>.pt``     -- the trained checkpoint, via the normal model store
"""

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "build" / "reference"
#: Batch size used for scoring only; larger than the training batch because
#: no gradients are held.
EVAL_BATCH = 512
#: Samples in the small probe that replaces the engine's in-training
#: evaluation. See :func:`_shrink_progress_evaluation`.
PROGRESS_SAMPLES = 256


@dataclass(frozen=True)
class Reference:
    """One published reference configuration."""

    name: str
    dataset: str
    topology: str
    epochs: int
    num_steps: int
    #: The engine always injects ``hidden``/``beta`` into the preset's
    #: parameters, so a row that wants a preset's own defaults has to restate
    #: them here -- otherwise every fully-connected preset trains as the same
    #: network and the table silently compares a topology against itself.
    hidden: int = 128
    beta: float = 0.5
    batch_size: int = 128
    #: Divisor, not a count: ``subset=1`` trains on the whole training split.
    subset: int = 1
    lr: float = 1e-2
    seed: int = 0
    note: str = ""
    topology_params: Dict[str, Any] = field(default_factory=dict)

    def command(self) -> str:
        """Return the one-liner that reproduces this row."""
        return (
            "python scripts/train_reference_models.py "
            f"--only {self.name}"
        )

    def engine_kwargs(self) -> Dict[str, Any]:
        """Return the TrainingEngine arguments for this configuration."""
        return {
            "dataset": self.dataset,
            "topology": self.topology,
            "hidden": self.hidden,
            "beta": self.beta,
            "epochs": self.epochs,
            "num_steps": self.num_steps,
            "batch_size": self.batch_size,
            "subset": self.subset,
            "lr": self.lr,
            "seed": self.seed,
            "device": "cpu",
            "topology_params": dict(self.topology_params),
        }


#: The published table. Kept small and honest on purpose: every row is a
#: shipped dataset on a shipped topology, trained from stock hyperparameters.
REFERENCES = (
    Reference(
        name="mnist-fc-legacy",
        dataset="mnist",
        topology="fc_legacy",
        epochs=3,
        num_steps=25,
        note="The default topology and the README quickstart's dataset.",
    ),
    Reference(
        name="mnist-fc-small",
        dataset="mnist",
        topology="fc_small",
        epochs=3,
        num_steps=25,
        hidden=32,
        beta=0.9,
        note="The smaller fully-connected preset, for edge-sized budgets.",
    ),
    Reference(
        name="mnist-conv-net",
        dataset="mnist",
        topology="conv_net",
        epochs=2,
        num_steps=25,
        note="Convolutional LIF; fewer epochs because each one costs more.",
    ),
    Reference(
        name="mnist-recurrent-net",
        dataset="mnist",
        topology="recurrent_net",
        epochs=3,
        num_steps=25,
        hidden=64,
        beta=0.9,
        note="Recurrent LIF with a one-step delayed feedback edge.",
    ),
    Reference(
        name="fashion-fc-legacy",
        dataset="fashion",
        topology="fc_legacy",
        epochs=3,
        num_steps=25,
        note="Fashion-MNIST: same shape, a harder ten-class problem.",
    ),
    Reference(
        name="dvs128-gesture-conv-net",
        dataset="dvs128_gesture",
        topology="conv_net",
        epochs=3,
        num_steps=25,
        # One bridged sample is 25 x 2 x 128 x 128 floats (~3.3 MB at fp32),
        # so the image rows' batch of 128 would be hundreds of megabytes of
        # input tensor alone before any activations.
        batch_size=16,
        # `num_classes` is deliberately absent: the engine injects the
        # registry's count (11), so stating it here would be a second place
        # for it to drift. `input_size` is sensor geometry, which the registry
        # does not carry -- a mismatch raises EventGeometryError by name.
        topology_params={"in_channels": 2, "input_size": 128},
        note=(
            "The event-native row: a real DVS recording rather than a "
            "re-rendered static image, on the canonical gesture task."
        ),
    ),
    Reference(
        name="kmnist-fc-legacy",
        dataset="kmnist",
        topology="fc_legacy",
        epochs=3,
        num_steps=25,
        note="Kuzushiji-MNIST: cursive Japanese characters.",
    ),
)


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
    torch.save(
        {
            "state_dict": engine.net.state_dict(),
            "meta": engine._meta(),
            "history": history[-1:],
            "manifest": engine._manifest(history[-1:]),
            "saved_at": time.time(),
        },
        path,
    )
    return path


def _shrink_progress_evaluation(engine: Any, dataset: str) -> None:
    """Make the engine's in-training progress check cheap.

    ``EvalMixin`` scores four batches of 1000 held-out samples every fifth
    step, which is right for a live dashboard and ruinous for a full-dataset
    run: it costs several times more than the training it reports on
    (``conv_net`` spent over an hour in it). The published number does not
    come from those checks anyway -- it comes from the full held-out pass in
    :func:`_full_test_accuracy` once training is done -- so the progress probe
    is shrunk to a single small batch. It matters more on the event path,
    where every probe sample is bridged rather than read from a tensor file.
    """
    from itertools import islice

    from spikeforge.data.data_loader import build_loader

    if _is_event(dataset):
        # Bridging is per-sample, so the probe is sized in samples directly
        # rather than by truncating a loader that does not exist here.
        from spikeforge.training.event_batches import batch_event_samples

        source = engine._test_source
        count = min(PROGRESS_SAMPLES, source.size())
        engine._test_batches = [
            batch_event_samples(
                source, engine._spec, range(count), engine._bridge
            )
        ]
        return
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


def _train(reference: Reference, out: Path) -> Dict[str, Any]:
    """Train one configuration, score it, and persist its checkpoint."""
    print(f"[{reference.name}] training ...", flush=True)
    started = time.perf_counter()
    engine = _engine_for(reference)
    _use_full_event_epoch(engine, reference.dataset)
    _shrink_progress_evaluation(engine, reference.dataset)
    history: List[Dict[str, Any]] = []
    last: Optional[Dict[str, Any]] = None
    for metrics in engine.train():
        history.append(metrics)
        last = metrics
    train_seconds = time.perf_counter() - started
    if last is None:
        raise RuntimeError(f"{reference.name}: training yielded no metrics")

    scored = _full_test_accuracy(engine, reference.dataset)
    checkpoint = _save_checkpoint(engine, reference, history, out)

    record = {
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
        "final_loss": round(float(last["loss"]), 4),
        "train_seconds": round(train_seconds, 1),
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "sha256": _digest(checkpoint),
        "command": reference.command(),
        **scored,
    }
    print(
        f"[{reference.name}] {record['test_accuracy']}% on "
        f"{record['test_samples']} held-out samples, "
        f"{record['train_seconds']}s",
        flush=True,
    )
    return record


def _markdown(records: List[Dict[str, Any]], hardware: Dict[str, Any]) -> str:
    """Render the results as the table the documentation page carries."""
    lines = [
        "| Configuration | Dataset | Topology | Test accuracy | Epochs | "
        "Steps | Train time |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| `{record['name']}` | {record['dataset']} | "
            f"`{record['topology']}` | **{record['test_accuracy']}%** | "
            f"{record['epochs']} | {record['num_steps']} | "
            f"{record['train_seconds']} s |"
        )
    lines.extend(
        [
            "",
            f"Hardware: {hardware['processor']}, "
            f"{hardware['torch_threads']} torch threads, CPU only. "
            f"Python {hardware['python']}, torch {hardware['torch']}.",
        ]
    )
    return "\n".join(lines)


#: Where a published checkpoint lands, and the catalog that points at it.
WEIGHTS_DIR = REPO_ROOT / "spikeforge_hub" / "weights"
CATALOG_PATH = REPO_ROOT / "spikeforge_hub" / "models.json"
#: Prefix for the catalog ids these checkpoints are published under.
CATALOG_PREFIX = "reference/"


def _catalog_entry(record: Dict[str, Any]) -> Dict[str, Any]:
    """Render one result record as a ``source: "reference"`` catalog entry.

    The dataset's licence and attribution are read from the provenance table
    that sits beside the dataset registry, never restated here: two copies of
    a licence is one copy that can go stale.
    """
    # Imported here, not at module scope: `spikeforge/__init__` pulls in
    # torch, and `--list` has to work without it.
    from spikeforge.data.dataset_provenance import dataset_provenance

    reference = next(
        item for item in REFERENCES if item.name == record["name"]
    )
    provenance = dataset_provenance(record["dataset"])
    return {
        "id": f"{CATALOG_PREFIX}{record['name']}",
        "name": (
            f"{record['dataset']} / {record['topology']} "
            f"({record['test_accuracy']}%)"
        ),
        # The checkpoint is a torch state dict for an snnTorch-backed LIF
        # network, so that is what it is labelled, not the NIR it can export.
        "framework": "snntorch",
        "kind": "state_dict",
        "source": "reference",
        # This project's own licence, for the weights. The training data's
        # own terms are a separate question and a separate field.
        "license": "BSD-3-Clause",
        "topology": record["topology"],
        "weights": f"{record['name']}.pt",
        "dataset": record["dataset"],
        "dataset_license": provenance.license,
        "dataset_attribution": provenance.attribution,
        "test_accuracy": record["test_accuracy"],
        "test_samples": record["test_samples"],
        "sha256": record["sha256"],
        "size_bytes": record["checkpoint_bytes"],
        "notes": (
            f"Trained by this project on the full {record['dataset']} "
            f"training split: {record['epochs']} epochs, "
            f"{record['num_steps']} time steps, seed {record['seed']}, CPU. "
            f"Scores {record['test_accuracy']}% on all "
            f"{record['test_samples']} held-out samples. A reference "
            f"configuration with stock hyperparameters, not a tuned attempt "
            f"at state of the art. Reproduce with: {record['command']}. "
            f"{reference.note}"
        ),
    }


def _sync_provenance() -> int:
    """Refresh every reference entry's dataset licence and attribution.

    These two fields are looked up from the provenance table by dataset name,
    not derived from the checkpoint, so they can be corrected without
    retraining — and they must be, because a licence gets verified (or
    changes upstream) long after the bytes were published. Everything that
    *does* describe the bytes (``sha256``, ``size_bytes``, ``test_accuracy``,
    ``test_samples``) is left untouched: republishing a checkpoint to fix a
    citation would replace the artifact the numbers were measured on.

    Returns the number of entries whose provenance changed.
    """
    from spikeforge.data.dataset_provenance import dataset_provenance

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    changed = 0
    for entry in catalog["entries"]:
        if entry.get("source") != "reference":
            continue
        provenance = dataset_provenance(str(entry["dataset"]))
        wanted = {
            "dataset_license": provenance.license,
            "dataset_attribution": provenance.attribution,
        }
        if all(entry.get(k) == v for k, v in wanted.items()):
            continue
        # Rebuild in order so the two fields sit beside the dataset they
        # describe rather than at the end of the record.
        rebuilt: Dict[str, Any] = {}
        for key, value in entry.items():
            rebuilt[key] = value
            if key == "dataset":
                rebuilt.update(wanted)
        if "dataset_license" not in rebuilt:
            rebuilt.update(wanted)
        entry.clear()
        entry.update(rebuilt)
        changed += 1
        print(
            f"provenance {entry['id']}: {provenance.license} "
            f"({provenance.source})"
        )
    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    print(f"catalog provenance synced: {changed} entries updated")
    return changed


def _publish(records: List[Dict[str, Any]], out: Path) -> None:
    """Copy the checkpoints into the hub package and update its catalog.

    Rewriting the catalog from the results is what keeps the pinned checksum,
    size, and accuracy true of the bytes that actually shipped -- three fields
    that are worse than useless when they drift.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for record in records:
        source = REPO_ROOT / record["checkpoint"]
        shutil.copy2(source, WEIGHTS_DIR / f"{record['name']}.pt")
        print(f"published {record['name']}.pt")

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    published = {record["name"] for record in records}
    kept = [
        entry
        for entry in catalog["entries"]
        if entry.get("source") != "reference"
        or entry["id"][len(CATALOG_PREFIX):] not in published
    ]
    catalog["entries"] = [_catalog_entry(r) for r in records] + kept
    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"catalog updated: {len(records)} trained entries, "
        f"{len(catalog['entries'])} total"
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Train the selected reference configurations and write the results."""
    parser = argparse.ArgumentParser(
        prog="train_reference_models",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="train only this configuration (repeatable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="directory for checkpoints and results",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch CPU threads; defaults to torch's own choice",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the configuration names and exit",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "copy the trained checkpoints into spikeforge_hub/weights/ and "
            "rewrite the catalog's reference entries to match"
        ),
    )
    parser.add_argument(
        "--sync-provenance",
        action="store_true",
        help=(
            "rewrite existing reference entries' dataset_license and "
            "dataset_attribution from the provenance table, without "
            "retraining anything, and exit"
        ),
    )
    args = parser.parse_args(argv)

    if args.sync_provenance:
        _sync_provenance()
        return 0

    if args.list:
        for reference in REFERENCES:
            print(f"{reference.name}\t{reference.dataset}\t"
                  f"{reference.topology}")
        return 0

    selected = [
        reference
        for reference in REFERENCES
        if args.only is None or reference.name in set(args.only)
    ]
    if not selected:
        print(f"no configuration matched {args.only}", file=sys.stderr)
        return 2

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hardware = _hardware()
    records = [_train(reference, out) for reference in selected]

    payload = {"hardware": hardware, "results": records}
    (out / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out / "results.md").write_text(
        _markdown(records, hardware) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {out / 'results.json'} and {out / 'results.md'}")
    if args.publish:
        _publish(records, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
