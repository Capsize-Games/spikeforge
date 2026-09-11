"""JSON-able payload for the event-driven energy WebSocket action.

The report reuses :func:`snn_targets.energy.accounting.account_spikes`, so
the socket surface matches the ``snn-energy`` CLI exactly. A dense-vs-sparse
comparison is attached because the parity check is meaningfully different from
the energy estimate; without a configured sample a deterministic synthetic
fixture is used, so the action is always answerable offline.
"""

from typing import Any, Dict, Optional

import torch

from server.target_payloads import shaped_spikes
from snn_interpreter.simulator.runner import run
from snn_interpreter.topology.spec import TopologySpec
from snn_targets.energy.accounting import (
    BATCH,
    DENSITY,
    SEED,
    STEPS,
    account_spikes,
)
from snn_targets.event_runtime.dense_compare import compare
from snn_targets.event_runtime.spike_view import synthetic_spikes


def _on_device(module: Any, spikes: torch.Tensor) -> torch.Tensor:
    """Move ``spikes`` onto the module's device for the sparse run."""
    param = next(module.parameters(), None)
    return spikes if param is None else spikes.to(param.device)


def _shaped(
    spec: TopologySpec, spikes: Optional[torch.Tensor]
) -> torch.Tensor:
    """Return the configured sample, or a synthetic fixture when absent."""
    if spikes is None:
        return synthetic_spikes(spec, STEPS, BATCH, SEED, DENSITY)
    return shaped_spikes(spikes, spec)


def energy_payload(
    spec: TopologySpec,
    module: Any,
    target: str,
    spikes: Optional[torch.Tensor] = None,
    sparse: bool = True,
) -> Dict[str, Any]:
    """Return the energy report plus a sparse-vs-dense comparison block."""
    shaped = _on_device(module, _shaped(spec, spikes))
    result, report = account_spikes(module, shaped, target, sparse)
    with torch.no_grad():
        dense = run(module, shaped)
    return {
        "target": target,
        "report": report.to_dict(),
        "comparison": compare(result, dense),
    }
