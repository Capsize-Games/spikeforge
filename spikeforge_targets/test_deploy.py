"""Run one trained topology across every simulator backend in one matrix.

The runner walks the registered targets, classifies the graph against each
target's declared capabilities, and — for the targets that have an executable
backend — compiles, runs, and compares the result to the reference interpreter
through the shared :func:`~spikeforge_targets.backends.compile_run` entry
point. Every target yields exactly one :class:`DeployCell`; nothing is dropped
and nothing is guessed at:

* the reference cell always runs in-process;
* an SDK-backed simulator whose SDK is absent reports ``available: false`` and
  a named reason (the ``compile_run`` contract);
* a declared-only simulator with no backend wired reports ``unavailable`` with
  a reason that says so, rather than being reported as a failure or skipped.

No simulator run is a device measurement, so the matrix carries
``estimate: true`` at the top level and every backend names its own ``path``.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from spikeforge_targets.backends import backend_for, compile_run
from spikeforge_targets.backends.result import (
    STATUS_OK,
    BackendResult,
)
from spikeforge_targets.capability_matrix import classify
from spikeforge_targets.registry import available, get_target, target_names
from spikeforge_targets.rewrite_drift import (
    READOUT_MAX_ABS,
    SPIKE_AGREEMENT,
    SPIKE_MAX_ABS,
)
from spikeforge_targets.target_spec import TargetSpec
from spikeforge_targets.test_deploy_result import (
    DeployCell,
    TestDeployMatrix,
)

#: The target every other backend is compared against.
DEFAULT_REFERENCE = "reference"
#: Note attached to every matrix so an emulator run is never read as measured.
ESTIMATE_NOTE = (
    "all simulator and emulator runs are estimates; no device was measured"
)
#: Name of the topology label used when the runner is handed a bare graph.
DEFAULT_TOPOLOGY = "graph"
#: Reason template for a declared target whose simulator is not wired yet.
NO_BACKEND_REASON = (
    "no simulator backend is registered for target {name!r}; it is declared "
    "only and installs with the {extra!r} extra"
)


def _parity_ok(parity: Optional[Mapping[str, Any]]) -> Optional[bool]:
    """Return the tolerance verdict for a comparison, or ``None``."""
    if not parity:
        return None
    readout = parity.get("readout") or {}
    spikes = parity.get("spikes") or {}
    return bool(
        readout.get("max_abs", 1.0) <= READOUT_MAX_ABS
        and spikes.get("max_abs", 1.0) <= SPIKE_MAX_ABS
        and spikes.get("agreement", 0.0) >= SPIKE_AGREEMENT
    )


def _reason(result: BackendResult) -> Optional[str]:
    """Return the named reason a result did not complete, or ``None``."""
    if result.status == STATUS_OK:
        return None
    return "; ".join(result.notes) or result.status


def _declared_only(
    spec: TargetSpec, available_flag: bool, capability: Dict[str, Any]
) -> DeployCell:
    """Return the honest cell for a target that has no wired backend."""
    reason = NO_BACKEND_REASON.format(name=spec.name, extra=spec.extra)
    return DeployCell(
        target=spec.name,
        kind=spec.kind,
        available=available_flag,
        backend=False,
        status="unavailable",
        reason=reason,
        capability=capability,
        notes=(reason,),
    )


def _cell(name: str, graph: Any, spikes: Any) -> DeployCell:
    """Return the test-deploy cell for ``name`` over a graph and spikes."""
    spec = get_target(name)
    available_flag = available(name)
    capability = classify(graph, spec).to_dict()
    backend = backend_for(name)
    if backend is None:
        return _declared_only(spec, available_flag, capability)
    result = compile_run(spec, graph, spikes)
    parity = result.compare if result.status == STATUS_OK else None
    return DeployCell(
        target=spec.name,
        kind=spec.kind,
        available=available_flag,
        backend=True,
        status=result.status,
        path=result.path,
        reason=_reason(result),
        parity=parity,
        parity_ok=_parity_ok(parity),
        capability=capability,
        notes=tuple(result.notes),
    )


def _selected(targets: Optional[Iterable[str]]) -> List[str]:
    """Return the target names to run, defaulting to the full registry."""
    return list(targets) if targets is not None else target_names()


def run_matrix(
    graph: Any,
    spikes: Any,
    topology: str = DEFAULT_TOPOLOGY,
    targets: Optional[Iterable[str]] = None,
) -> TestDeployMatrix:
    """Test-deploy ``graph`` on every registered simulator; never raise.

    ``graph`` is a target-ready NIR graph and ``spikes`` its ``[T, ...]``
    input. Each target yields one cell; an absent SDK or a declared-only
    simulator is reported ``unavailable`` with a named reason, an available
    backend that refuses the graph is reported ``error`` with the reason, and
    a completed run carries its reference parity comparison.
    """
    cells: Tuple[DeployCell, ...] = tuple(
        _cell(name, graph, spikes) for name in _selected(targets)
    )
    return TestDeployMatrix(
        topology=topology,
        steps=int(spikes.size(0)),
        reference=DEFAULT_REFERENCE,
        estimate=True,
        cells=cells,
        notes=(ESTIMATE_NOTE,),
    )


def matrix_by_name(
    topology: str,
    steps: int,
    batch: int,
    seed: int,
    targets: Optional[Iterable[str]] = None,
) -> TestDeployMatrix:
    """Test-deploy a shipped topology's graph on a deterministic fixture.

    Imported lazily so the graph-based :func:`run_matrix` stays free of the
    fixture/registry machinery a pure caller never needs.
    """
    from spikeforge.cli import fixture
    from spikeforge.nir_bridge.exporter import to_nir

    spec, module, spikes = fixture.synthetic_input(
        topology, steps, batch, seed
    )
    return run_matrix(
        to_nir(spec, module),
        spikes,
        topology=topology,
        targets=targets,
    )
