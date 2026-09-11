"""JSON-able payloads for the deployment-target WebSocket actions.

Mirrors :mod:`server.payloads`: the report reuses
:func:`snn_targets.report.deployment_report` and the list reuses
the availability-annotated registry summary, so the socket surface matches the
``verify`` CLI exactly. Validation is attached only when a shaped spike input
is supplied; the drift check is meaningfully different from the capability
classification, so it is never invented when the inputs are absent.
"""

from typing import Any, Dict, Optional

import torch

from snn_interpreter.simulator import input_shape
from snn_interpreter.topology.spec import TopologySpec
from snn_targets.report import deployment_report
from snn_targets.summary import target_summaries


def target_list_payload() -> Dict[str, Any]:
    """Return the availability-annotated target registry."""
    return {"targets": target_summaries()}


def shaped_spikes(spikes: torch.Tensor, spec: TopologySpec) -> torch.Tensor:
    """Return ``spikes`` reshaped to ``spec``'s input-stage layout."""
    return input_shape.to_input_shape(spikes, spec)


def _on_device(module: Any, spikes: torch.Tensor) -> torch.Tensor:
    """Move ``spikes`` onto the module's device for the validator."""
    param = next(module.parameters(), None)
    return spikes if param is None else spikes.to(param.device)


def deployment_report_payload(
    spec: TopologySpec,
    module: Any,
    target: str,
    spikes: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Return a deployment report, attaching validation when ``spikes`` exist.

    ``spikes`` must already be shaped for ``spec``; without them the report
    carries ``validation: None`` rather than a fabricated drift section.
    """
    if spikes is None:
        return deployment_report(spec, target)
    return deployment_report(
        spec, target, module=module, spikes=_on_device(module, spikes)
    )
