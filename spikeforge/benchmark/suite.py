"""Run a small benchmark suite and record it with provenance metadata.

:func:`run_suite` is the CI-sized entry point: it benchmarks a set of
topologies (optionally restricting the modes), attaches the Phase 6a library
versions and a timestamp, and persists the record through a
:class:`~spikeforge.benchmark.store.BenchmarkStore`.
"""

import time
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from spikeforge.benchmark.config import BenchmarkConfig, default_config
from spikeforge.benchmark.harness import run_benchmark
from spikeforge.benchmark.store import BenchmarkStore
from spikeforge.tracking.versions import library_versions

#: Topologies measured by default; small enough to run in CI.
DEFAULT_TOPOLOGIES = ("fc_small", "conv_net")
#: Extra mode label produced only when ``compiled`` is enabled.
COMPILED_MODE = "production_compiled"


def suite_config(
    topologies: Sequence[str],
    modes: Optional[Sequence[str]] = None,
    base: Optional[BenchmarkConfig] = None,
) -> BenchmarkConfig:
    """Return a tiny config measuring ``topologies`` (and ``modes``)."""
    config = base if base is not None else default_config()
    compiled = modes is not None and COMPILED_MODE in modes
    return replace(
        config, topologies=tuple(topologies), compiled=compiled
    )


def _select(
    results: List[Dict[str, Any]], modes: Optional[Sequence[str]]
) -> List[Dict[str, Any]]:
    """Keep only the records whose mode is in ``modes`` (all when None)."""
    if modes is None:
        return results
    wanted = set(modes)
    return [record for record in results if record["mode"] in wanted]


def with_metadata(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the run record with a timestamp and version provenance."""
    enriched = dict(record)
    enriched["created_at"] = time.time()
    enriched["versions"] = library_versions()
    return enriched


def run_suite(
    topologies: Sequence[str] = DEFAULT_TOPOLOGIES,
    modes: Optional[Sequence[str]] = None,
    config: Optional[BenchmarkConfig] = None,
    store: Optional[BenchmarkStore] = None,
    label: Optional[str] = "suite",
    save: bool = True,
) -> Dict[str, Any]:
    """Benchmark ``topologies`` and optionally persist the record.

    Returns the same report shape as :func:`run_benchmark`, plus
    ``created_at``, ``versions``, and (when ``save``) a ``run_id``.
    """
    report = run_benchmark(suite_config(topologies, modes, config))
    report["results"] = _select(report["results"], modes)
    enriched = with_metadata(report)
    if save:
        target = store if store is not None else BenchmarkStore()
        enriched["run_id"] = target.save(enriched, label=label)
    return enriched
