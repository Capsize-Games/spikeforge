"""Map operation counts to a target's declared energy/latency estimate.

The counts come from the event-driven runtime; this layer only turns them into
an honest report. A target with a bundled cost table yields an estimate
(``estimate: true``), a target with none reports ``basis: "unavailable"`` and
``None`` numbers, and a device-reported measurement flips the report to
``estimate: false``. Nothing is ever presented as measured without a device.
"""

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from spikeforge.topology.registry import build_topology
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule
from spikeforge_targets.energy import probe, target_costs
from spikeforge_targets.energy.report import EnergyReport
from spikeforge_targets.event_runtime.sparse_result import SparseResult
from spikeforge_targets.event_runtime.sparse_runner import sparse_run
from spikeforge_targets.event_runtime.spike_view import synthetic_spikes

#: ``basis`` values naming where a report's numbers come from.
BASIS_TABLE = "declared cost table"
BASIS_UNAVAILABLE = "unavailable"
BASIS_MEASURED = "device measurement"

#: Default synthetic fixture for a topology-only account.
STEPS = 8
BATCH = 2
SEED = 0
DENSITY = 0.1

_Fixture = Tuple[TopologySpec, StageModule, torch.Tensor]


def _counts(source: Any) -> Dict[str, int]:
    """Return a raw counts mapping from a ``SparseResult`` or a mapping."""
    recorded = getattr(source, "counts", None)
    return dict(recorded if recorded is not None else source)


def _ops(counts: Mapping[str, int]) -> Dict[str, int]:
    """Return the flat SOP/MAC/AC operation counts."""
    return {
        "sop": int(counts.get("sop", 0)),
        "mac": int(counts.get("mac", 0)),
        "ac": int(counts.get("ac", 0)),
    }


def _timesteps(counts: Mapping[str, int]) -> Optional[int]:
    """Return the recorded timestep count, or ``None`` when absent."""
    value = counts.get("timesteps")
    return None if value is None else int(value)


def _efficiency(ops: Mapping[str, int]) -> Dict[str, Optional[float]]:
    """Return the sparse efficiency ratio ``sop / mac``."""
    mac = ops["mac"]
    return {"sop_over_mac": None if mac == 0 else ops["sop"] / mac}


def _estimate(counts: Mapping[str, int], table: Any) -> EnergyReport:
    """Return a declared-cost estimate report for ``counts``."""
    ops = _ops(counts)
    steps = _timesteps(counts) or 0
    return EnergyReport(
        target=table.target,
        estimate=True,
        basis=BASIS_TABLE,
        timesteps=_timesteps(counts),
        ops=ops,
        efficiency=_efficiency(ops),
        energy=table.energy_pj(ops["sop"], ops["mac"], ops["ac"]),
        latency=table.latency_ns(steps),
        notes=(
            "estimate only; no device measured",
            f"declared cost source: {table.source}",
        ),
    )


def _unavailable(counts: Mapping[str, int], target: str) -> EnergyReport:
    """Return an honest report for a target without a declared table."""
    ops = _ops(counts)
    return EnergyReport(
        target=target,
        estimate=True,
        basis=BASIS_UNAVAILABLE,
        timesteps=_timesteps(counts),
        ops=ops,
        efficiency=_efficiency(ops),
        energy=None,
        latency=None,
        notes=(
            f"no declared cost table for target {target!r}",
            "energy and latency are unavailable, not estimated",
            probe.reason(target),
        ),
    )


def _measured(
    counts: Mapping[str, int],
    target: str,
    measurement: Mapping[str, Any],
) -> EnergyReport:
    """Return a report built from a device's own measured timing."""
    ops = _ops(counts)
    device = str(measurement.get("device", target))
    return EnergyReport(
        target=target,
        estimate=False,
        basis=BASIS_MEASURED,
        timesteps=_timesteps(counts),
        ops=ops,
        efficiency=_efficiency(ops),
        energy=_measured_energy(measurement),
        latency=_measured_latency(measurement),
        notes=(f"measured on {device}",),
        measured=dict(measurement),
    )


def _measured_energy(
    measurement: Mapping[str, Any]
) -> Optional[Dict[str, float]]:
    """Return the measured energy block, or ``None`` when it is absent."""
    if "total_pj" not in measurement:
        return None
    return {"total_pj": float(measurement["total_pj"])}


def _measured_latency(
    measurement: Mapping[str, Any]
) -> Optional[Dict[str, float]]:
    """Return the measured latency block, or ``None`` when it is absent."""
    if "total_ns" not in measurement:
        return None
    return {"total_ns": float(measurement["total_ns"])}


def account(
    source: Any,
    target: str,
    measurement: Optional[Mapping[str, Any]] = None,
) -> EnergyReport:
    """Return the energy/latency report for ``source`` against ``target``.

    ``source`` is a
    :class:`~spikeforge_targets.event_runtime.SparseResult` or a raw counts
    mapping. A ``measurement`` (or one a device probe reports) makes
    the report measured; otherwise a declared cost table yields an estimate and
    a target without one reports ``basis: "unavailable"``.
    """
    counts = _counts(source)
    found = measurement if measurement is not None else probe.measure(target)
    if found is not None:
        return _measured(counts, target, found)
    table = target_costs.load(target)
    if table is None:
        return _unavailable(counts, target)
    return _estimate(counts, table)


def topology_fixture(
    topology: str,
    steps: int = STEPS,
    batch: int = BATCH,
    seed: int = SEED,
) -> _Fixture:
    """Return ``(spec, module, spikes)`` for a deterministic fixture."""
    spec, module = build_topology(topology)
    spikes = synthetic_spikes(spec, steps, batch, seed, DENSITY)
    return spec, module, spikes


def account_spikes(
    module: StageModule,
    spikes: torch.Tensor,
    target: str,
    sparse: bool = True,
) -> Tuple[SparseResult, EnergyReport]:
    """Run ``module``'s event-driven path and account it for ``target``.

    With ``sparse`` off the report uses the dense baseline (``sop == mac``),
    so the same call expresses both the worst case and the event-driven view.
    """
    with torch.no_grad():
        result = sparse_run(module, spikes)
    counts = dict(result.counts)
    if not sparse:
        counts["sop"] = counts["mac"]
    return result, account(counts, target)


def measure_topology(
    topology: str,
    target: str,
    steps: int = STEPS,
    batch: int = BATCH,
    seed: int = SEED,
    sparse: bool = True,
) -> Tuple[SparseResult, EnergyReport]:
    """Build ``topology``'s fixture and account its event-driven run."""
    spec, module, spikes = topology_fixture(topology, steps, batch, seed)
    return account_spikes(module, spikes, target, sparse)
