"""Opt-in event-driven energy block for the benchmark harness.

Kept out of :mod:`spikeforge.benchmark.harness` so that module stays
within the project's file-length limit; the harness calls this only when its
config asks for energy, and the returned block is additive on the record.
"""

from typing import Any, Dict, Optional

import torch

from spikeforge.benchmark.config import BenchmarkConfig
from spikeforge.targets_extra import require as require_targets
from spikeforge.topology.stage_module import StageModule


def energy_block(
    module: StageModule, spikes: torch.Tensor, config: BenchmarkConfig
) -> Optional[Dict[str, Any]]:
    """Return the opt-in event-driven energy estimate, or None.

    ``spikeforge-targets`` is only imported here, once energy accounting is
    actually requested, so a plain ``spikeforge`` install keeps every other
    benchmark path working.
    """
    if not config.energy:
        return None
    accounting = require_targets("spikeforge_targets.energy.accounting")
    sparse_runner = require_targets(
        "spikeforge_targets.event_runtime.sparse_runner"
    )
    with torch.no_grad():
        result = sparse_runner.sparse_run(module, spikes)
    report = accounting.account(result, config.energy_target)
    return {
        "report": report.to_dict(),
        "counts": dict(result.counts),
        "density": result.density,
    }
