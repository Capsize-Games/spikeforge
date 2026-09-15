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
from typing import Any, Dict, List, Optional, Tuple

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
        name="ssc-fc-legacy",
        dataset="ssc",
        topology="fc_legacy",
        epochs=3,
        num_steps=25,
        batch_size=128,
        # A cochlea has 700 channels and no second spatial axis, so the flat
        # sensor area is 700. `num_classes` is absent on purpose: the engine
        # injects the registry's 35.
        topology_params={"input_size": 700},
        note=(
            "The audio row: a spiking cochlea model of Speech Commands, so "
            "the catalog spans static vision, event vision, and audio."
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


def _markdown_row(record: Dict[str, Any]) -> str:
    """Return one Markdown table row for a trained config's result."""
    return (
        f"| `{record['name']}` | {record['dataset']} | "
        f"`{record['topology']}` | **{record['test_accuracy']}%** | "
        f"{record['epochs']} | {record['num_steps']} | "
        f"{record['train_seconds']} s |"
    )


def _markdown_footer(hardware: Dict[str, Any]) -> str:
    """Return the hardware footer line for the results table."""
    return (
        f"Hardware: {hardware['processor']}, "
        f"{hardware['torch_threads']} torch threads, CPU only. "
        f"Python {hardware['python']}, torch {hardware['torch']}."
    )


def _markdown(records: List[Dict[str, Any]], hardware: Dict[str, Any]) -> str:
    """Render the results as the table the documentation page carries."""
    lines = [
        "| Configuration | Dataset | Topology | Test accuracy | Epochs | "
        "Steps | Train time |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(_markdown_row(record) for record in records)
    lines.extend(["", _markdown_footer(hardware)])
    return "\n".join(lines)


#: Where a published checkpoint lands, and the catalog that points at it.
WEIGHTS_DIR = REPO_ROOT / "spikeforge_hub" / "weights"
CATALOG_PATH = REPO_ROOT / "spikeforge_hub" / "models.json"
#: Prefix for the catalog ids these checkpoints are published under.
CATALOG_PREFIX = "reference/"


def _catalog_identity(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the identity/labelling fields of a catalog entry."""
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
    }


def _catalog_scoring(
    record: Dict[str, Any], provenance: Any
) -> Dict[str, Any]:
    """Return the provenance/scoring fields of a catalog entry."""
    return {
        "dataset_license": provenance.license,
        "dataset_attribution": provenance.attribution,
        "test_accuracy": record["test_accuracy"],
        "test_samples": record["test_samples"],
        "sha256": record["sha256"],
        "size_bytes": record["checkpoint_bytes"],
    }


def _catalog_fields(
    record: Dict[str, Any], provenance: Any
) -> Dict[str, Any]:
    """Return the catalog entry's fields other than its ``notes``."""
    fields = _catalog_identity(record)
    fields.update(_catalog_scoring(record, provenance))
    return fields


def _catalog_notes(record: Dict[str, Any], reference: Reference) -> str:
    """Return the human-readable reproduction note for a catalog entry."""
    return (
        f"Trained by this project on the full {record['dataset']} "
        f"training split: {record['epochs']} epochs, "
        f"{record['num_steps']} time steps, seed {record['seed']}, CPU. "
        f"Scores {record['test_accuracy']}% on all "
        f"{record['test_samples']} held-out samples. A reference "
        f"configuration with stock hyperparameters, not a tuned attempt "
        f"at state of the art. Reproduce with: {record['command']}. "
        f"{reference.note}"
    )


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
    entry = _catalog_fields(record, provenance)
    entry["notes"] = _catalog_notes(record, reference)
    return entry


def _rebuild_with_provenance(
    entry: Dict[str, Any], wanted: Dict[str, str]
) -> Dict[str, Any]:
    """Return ``entry`` with ``wanted`` fields placed beside ``dataset``.

    Rebuilt in order so the two fields sit beside the dataset they describe
    rather than at the end of the record.
    """
    rebuilt: Dict[str, Any] = {}
    for key, value in entry.items():
        rebuilt[key] = value
        if key == "dataset":
            rebuilt.update(wanted)
    if "dataset_license" not in rebuilt:
        rebuilt.update(wanted)
    return rebuilt


def _print_synced_entry(entry: Dict[str, Any], provenance: Any) -> None:
    """Print the provenance change for one synced catalog entry."""
    print(
        f"provenance {entry['id']}: {provenance.license} "
        f"({provenance.source})"
    )


def _sync_entry(entry: Dict[str, Any]) -> bool:
    """Refresh one catalog entry's provenance; report if it changed."""
    from spikeforge.data.dataset_provenance import dataset_provenance

    if entry.get("source") != "reference":
        return False
    provenance = dataset_provenance(str(entry["dataset"]))
    wanted = {
        "dataset_license": provenance.license,
        "dataset_attribution": provenance.attribution,
    }
    if all(entry.get(k) == v for k, v in wanted.items()):
        return False
    rebuilt = _rebuild_with_provenance(entry, wanted)
    entry.clear()
    entry.update(rebuilt)
    _print_synced_entry(entry, provenance)
    return True


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
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    changed = sum(_sync_entry(entry) for entry in catalog["entries"])
    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    print(f"catalog provenance synced: {changed} entries updated")
    return changed


def _copy_checkpoints(records: List[Dict[str, Any]]) -> None:
    """Copy every trained checkpoint into the hub's weights directory."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for record in records:
        source = REPO_ROOT / record["checkpoint"]
        shutil.copy2(source, WEIGHTS_DIR / f"{record['name']}.pt")
        print(f"published {record['name']}.pt")


def _rewrite_catalog(records: List[Dict[str, Any]]) -> None:
    """Replace published reference entries with freshly built ones."""
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


def _publish(records: List[Dict[str, Any]], out: Path) -> None:
    """Copy the checkpoints into the hub package and update its catalog.

    Rewriting the catalog from the results is what keeps the pinned checksum,
    size, and accuracy true of the bytes that actually shipped -- three fields
    that are worse than useless when they drift.
    """
    _copy_checkpoints(records)
    _rewrite_catalog(records)


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags that pick which configs to run and how."""
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


def _add_action_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags that select a mode instead of training normally."""
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


def _add_sync_provenance_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--sync-provenance`` maintenance-mode flag."""
    parser.add_argument(
        "--sync-provenance",
        action="store_true",
        help=(
            "rewrite existing reference entries' dataset_license and "
            "dataset_attribution from the provenance table, without "
            "retraining anything, and exit"
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="train_reference_models",
        description=__doc__.splitlines()[0],
    )
    _add_selection_args(parser)
    _add_action_args(parser)
    _add_sync_provenance_arg(parser)
    return parser


def _print_config_names() -> None:
    """Print each reference configuration's name, dataset, and topology."""
    for reference in REFERENCES:
        print(
            f"{reference.name}\t{reference.dataset}\t{reference.topology}"
        )


def _select_references(only: Optional[List[str]]) -> List[Reference]:
    """Return the reference configs matching ``--only``, or all of them."""
    return [
        reference
        for reference in REFERENCES
        if only is None or reference.name in set(only)
    ]


def _write_results(
    records: List[Dict[str, Any]], hardware: Dict[str, Any], out: Path
) -> None:
    """Write ``results.json`` and ``results.md`` under ``out``."""
    payload = {"hardware": hardware, "results": records}
    (out / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out / "results.md").write_text(
        _markdown(records, hardware) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {out / 'results.json'} and {out / 'results.md'}")


def _train_and_write(
    args: argparse.Namespace, selected: List[Reference]
) -> int:
    """Train every selected config, write results, and optionally publish."""
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hardware = _hardware()
    records = [_train(reference, out) for reference in selected]
    _write_results(records, hardware, out)
    if args.publish:
        _publish(records, out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Train the selected reference configurations and write the results."""
    args = _build_parser().parse_args(argv)
    if args.sync_provenance:
        _sync_provenance()
        return 0
    if args.list:
        _print_config_names()
        return 0
    selected = _select_references(args.only)
    if not selected:
        print(f"no configuration matched {args.only}", file=sys.stderr)
        return 2
    return _train_and_write(args, selected)


if __name__ == "__main__":
    sys.exit(main())
