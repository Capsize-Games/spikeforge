"""Opt-in event-driven energy block for the benchmark harness.

Kept out of :mod:`snn_interpreter.benchmark.harness` so that module stays
within the project's file-length limit; the harness calls this only when its
config asks for energy, and the returned block is additive on the record.
"""

from typing import Any, Dict, Optional

import torch

from snn_interpreter.benchmark.config import BenchmarkConfig
from snn_interpreter.topology.stage_module import StageModule
from snn_targets.energy.accounting import account
from snn_targets.event_runtime.sparse_runner import sparse_run


def energy_block(
    module: StageModule, spikes: torch.Tensor, config: BenchmarkConfig
) -> Optional[Dict[str, Any]]:
    """Return the opt-in event-driven energy estimate, or None."""
    if not config.energy:
        return None
    with torch.no_grad():
        result = sparse_run(module, spikes)
    report = account(result, config.energy_target)
    return {
        "report": report.to_dict(),
        "counts": dict(result.counts),
        "density": result.density,
    }
