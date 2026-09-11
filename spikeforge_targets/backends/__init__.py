
"""Executable deployment backends behind one isolated SDK probe.

:func:`compile_run` is the single entry point: it applies the target's
declared substitutions, gates on the backend's availability and on the
rewrite being complete, then compiles, runs, and compares the result to the
reference interpreter. Every outcome is a :class:`BackendResult` with an
honest status; nothing raises and nothing is faked.
"""

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from spikeforge_targets.backends.base import Backend
from spikeforge_targets.backends.compare import compare_results
from spikeforge_targets.backends.lava_backend import LavaBackend
from spikeforge_targets.backends.norse_backend import NorseBackend
from spikeforge_targets.backends.reference_backend import (
    ReferenceBackend,
)
from spikeforge_targets.backends.result import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    BackendResult,
)
from spikeforge_targets.quantize import quantize
from spikeforge_targets.quantize_result import QuantizationResult
from spikeforge_targets.registry import get_target
from spikeforge_targets.rewrite import rewrite
from spikeforge_targets.rewrite_result import RewriteResult

__all__ = [
    "BACKENDS",
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_UNAVAILABLE",
    "BackendResult",
    "backend_for",
    "compile_run",
]

#: The registered backend for every executable target kind.
BACKENDS: Dict[str, Backend] = {
    "reference": ReferenceBackend(),
    "norse": NorseBackend(),
    "lava_loihi2": LavaBackend(),
}


def backend_for(name: str) -> Optional[Backend]:
    """Return the backend registered for ``name``, or ``None``."""
    return BACKENDS.get(name)


def _unavailable(
    spec: Any, rewritten: RewriteResult, quantized: QuantizationResult
) -> BackendResult:
    """Return the honest result for a backend whose SDK is absent."""
    extra = spec.extra or "backend"
    note = f"install the {extra!r} extra to run the {spec.name!r} target"
    return BackendResult(
        spec.name,
        STATUS_UNAVAILABLE,
        notes=(note,),
        rewritten=rewritten.report.to_dict(),
        quantization=quantized.report.to_dict(),
    )


def _refusal_notes(report: Any) -> Tuple[str, ...]:
    """Return the named reasons a rewritten graph is not target-ready."""
    notes = [
        f"unfixable node {item['node']} ({item['primitive']})"
        for item in report.unfixable
    ]
    notes += [
        f"skipped node {item['node']} ({item['from']}->{item['to']})"
        for item in report.skipped
    ]
    return tuple(notes) or ("graph is not target-ready",)


def _refused(
    rewritten: RewriteResult, quantized: QuantizationResult
) -> BackendResult:
    """Return the honest result for a graph the target cannot run."""
    return BackendResult(
        rewritten.report.target,
        STATUS_ERROR,
        notes=_refusal_notes(rewritten.report),
        rewritten=rewritten.report.to_dict(),
        quantization=quantized.report.to_dict(),
    )


def _finalise(
    result: BackendResult,
    rewritten: RewriteResult,
    quantized: QuantizationResult,
    spikes: Any,
) -> BackendResult:
    """Attach the rewrite/quantization reports and the reference comparison."""
    reference = ReferenceBackend().run(quantized.graph, spikes)
    compare = compare_results(result, reference)
    return replace(
        result,
        rewritten=rewritten.report.to_dict(),
        compare=compare,
        quantization=quantized.report.to_dict(),
    )


def _attempt(
    spec: Any,
    backend: Backend,
    rewritten: RewriteResult,
    quantized: QuantizationResult,
    spikes: Any,
) -> BackendResult:
    """Compile, run, and compare, folding any failure into a status."""
    try:
        compiled = backend.compile(quantized.graph, spec)
        result = backend.run(compiled, spikes)
        return _finalise(result, rewritten, quantized, spikes)
    except Exception as exc:
        note = f"{type(exc).__name__}: {exc}"
        return BackendResult(
            spec.name,
            STATUS_ERROR,
            notes=(note,),
            rewritten=rewritten.report.to_dict(),
            quantization=quantized.report.to_dict(),
        )


def compile_run(
    target: Any, graph_or_spec: Any, spikes: Any
) -> BackendResult:
    """Compile and run ``graph_or_spec`` on ``target``; never raise.

    ``target`` is a target name or :class:`TargetSpec`. An absent SDK yields
    ``unavailable``, an unfixable or skipped node yields ``error``, and a
    completed run yields ``ok`` with the rewrite report and the reference
    comparison attached.
    """
    spec = get_target(target) if isinstance(target, str) else target
    backend = backend_for(spec.name)
    if backend is None:
        note = f"no executable backend is registered for {spec.name!r}"
        return BackendResult(spec.name, STATUS_ERROR, notes=(note,))
    rewritten = rewrite(graph_or_spec, spec, spikes)
    quantized = quantize(rewritten.graph, spec, spikes)
    if not backend.available():
        return _unavailable(spec, rewritten, quantized)
    if not rewritten.report.ready():
        return _refused(rewritten, quantized)
    return _attempt(spec, backend, rewritten, quantized, spikes)
