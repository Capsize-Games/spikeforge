"""JSON-able payloads for the backend-run WebSocket action.

The payload is the :meth:`BackendResult.to_dict` of an executed (or refused)
backend run. A missing sample or an export failure is reported as an honest
``error`` result with a named reason rather than a raised exception, matching
the deployment-report handler's degrade-don't-raise contract.
"""

from typing import Any, Dict, Optional

import torch

from snn_interpreter.nir_bridge.exporter import to_nir
from snn_interpreter.topology.spec import TopologySpec
from snn_targets.backends import STATUS_ERROR, compile_run
from snn_targets.backends.result import BackendResult

#: Reason reported when no shaped sample is available to run.
NO_SAMPLE = "no sample is loaded; select a sample before running a backend"


def _device_input(module: Any, spikes: torch.Tensor) -> torch.Tensor:
    """Move ``spikes`` onto the module's device for the backend run."""
    param = next(module.parameters(), None)
    return spikes if param is None else spikes.to(param.device)


def _error(target: str, reason: str) -> Dict[str, Any]:
    """Return an error-shaped backend payload with a named reason."""
    return BackendResult(
        target=target, status=STATUS_ERROR, notes=(reason,)
    ).to_dict()


def backend_run_payload(
    spec: TopologySpec,
    module: Any,
    target: str,
    spikes: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Return the backend run payload for a spec, module, and shaped spikes.

    ``spikes`` must already be shaped for ``spec``; without them the payload
    is an honest error rather than a fabricated run.
    """
    if spikes is None:
        return _error(target, NO_SAMPLE)
    try:
        graph = to_nir(spec, module)
        result = compile_run(target, graph, _device_input(module, spikes))
    except Exception as exc:
        return _error(target, f"{type(exc).__name__}: {exc}")
    return result.to_dict()
