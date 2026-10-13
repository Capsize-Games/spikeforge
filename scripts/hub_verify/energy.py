"""Produce the SOP/MAC/AC energy/latency report for every known target.

Reuses :func:`spikeforge_targets.energy.accounting.account_spikes` exactly
as elsewhere in the project; a target whose SDK is not installed on this
runner reports ``basis: "unavailable"`` rather than a fabricated number --
the same honest-degradation convention ``ci.yml``'s ``test-deploy`` job
checks for the curated catalog. Energy is descriptive here, never a pass
gate: an artifact is not rejected for costing more than some threshold.
"""

from typing import Any, Dict, List

from hub_verify.fixtures import probe_spikes
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule
from spikeforge_targets import registry
from spikeforge_targets.energy.accounting import account_spikes


def energy_reports(
    spec: TopologySpec, module: StageModule
) -> List[Dict[str, Any]]:
    """Return one energy/latency report per registered target."""
    spikes = probe_spikes(spec)
    reports = []
    for name in registry.target_names():
        _, report = account_spikes(module, spikes, name)
        reports.append(report.to_dict())
    return reports
