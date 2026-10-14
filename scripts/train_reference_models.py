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
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from train_reference_engine import (
    PROGRESS_SAMPLES as PROGRESS_SAMPLES,
)
from train_reference_engine import (
    _engine_for as _engine_for,
)
from train_reference_engine import (
    _is_event as _is_event,
)
from train_reference_engine import (
    _use_full_event_epoch as _use_full_event_epoch,
)
from train_reference_report import _markdown
from train_reference_score import (
    _event_test_batches as _event_test_batches,
)
from train_reference_score import (
    _full_test_accuracy as _full_test_accuracy,
)
from train_reference_score import (
    _hardware,
    _train,
)
from train_reference_score import (
    _shrink_progress_evaluation as _shrink_progress_evaluation,
)
from train_reference_sync import _publish, _sync_provenance
from train_reference_table import (
    DEFAULT_OUT,
)
from train_reference_table import (
    REFERENCES as REFERENCES,
)
from train_reference_table import (
    Reference as Reference,
)


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
    main(sys.argv[1:])
