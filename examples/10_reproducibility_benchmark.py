"""Record a reproducibility manifest and save/compare a benchmark run.

Two independent reproducibility surfaces are shown: a
``ReproducibilityManifest`` whose ``config_hash`` is order-independent by
construction, and the ``BenchmarkStore`` used to save a run and diff it
against itself (a self-comparison reports every metric as unchanged).

Run from the repository root::

    venv/bin/python examples/10_reproducibility_benchmark.py

The manifest hash covers configuration only; it makes a run reproducible,
not bit-exact (CUDA kernels and thread scheduling stay outside the
process's control). The benchmark writes into a temporary directory.
"""

import tempfile
from typing import Any, Dict

from snn_interpreter.benchmark import (
    BenchmarkConfig,
    BenchmarkStore,
    compare_runs,
    run_benchmark,
    with_metadata,
)
from snn_interpreter.tracking.config_hash import config_hash
from snn_interpreter.tracking.manifest import ReproducibilityManifest

#: Example run configuration, deliberately written in two key orders.
CONFIG: Dict[str, Any] = {
    "dataset": "mnist",
    "topology": "fc_small",
    "hidden": 32,
    "seed": 0,
}


def show_manifest() -> None:
    """Print an order-independent hash and the reproducible block."""
    reordered = {
        "seed": 0,
        "hidden": 32,
        "topology": "fc_small",
        "dataset": "mnist",
    }
    manifest = ReproducibilityManifest(CONFIG, seed=0)
    print("config_hash stable:", config_hash(CONFIG) == config_hash(reordered))
    print("manifest hash:", manifest.config_hash)
    print("bit_exact:", manifest.to_dict()["reproducible"]["bit_exact"])


def save_and_compare(directory: str) -> None:
    """Save a benchmark run and compare it against itself."""
    config = BenchmarkConfig(topologies=("fc_small",), steps=4, repeats=1)
    record = with_metadata(run_benchmark(config))
    store = BenchmarkStore(directory)
    run_id = store.save(record, label="example")
    print("saved benchmark run:", run_id)
    result = compare_runs(store.load(run_id), store.load(run_id), 0.1)
    print("compared cases:", result["compared"])


def main() -> int:
    """Show the manifest and the benchmark save/compare journey."""
    show_manifest()
    with tempfile.TemporaryDirectory() as directory:
        save_and_compare(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
