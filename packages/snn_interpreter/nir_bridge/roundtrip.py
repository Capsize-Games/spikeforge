"""Export, persist, reload and re-interpret a topology's graph.

This is the cross-library fidelity guarantee: the report says whether the
graph that was written to disk reproduces, bit for bit, the interpretation of
the in-memory export of ``spec``/``module``. Metrics are the plain numbers
from :mod:`snn_interpreter.nir_bridge.drift`, so the report is JSON-able.
"""

import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional

import torch

from snn_interpreter.nir_bridge import drift
from snn_interpreter.nir_bridge.exporter import to_nir
from snn_interpreter.nir_bridge.interpreter import NirInterpreter
from snn_interpreter.nir_bridge.serialization import load_graph, save_graph
from snn_interpreter.topology.spec import TopologySpec

#: A per-trace mapping of metric name to plain float.
Metrics = Dict[str, Dict[str, float]]


@contextmanager
def _target_path(path: Optional[str]) -> Iterator[str]:
    """Yield ``path``, or a temporary file removed when the block exits."""
    if path is not None:
        yield path
        return
    handle, temporary = tempfile.mkstemp(suffix=".nir.json")
    os.close(handle)
    try:
        yield temporary
    finally:
        os.remove(temporary)


def _trace_metrics(
    actual: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> Metrics:
    """Return drift metrics for every trace name in ``reference``."""
    return {
        name: drift.compare(actual[name], reference[name])
        for name in reference
    }


def _exact(metrics: Mapping[str, Mapping[str, float]]) -> bool:
    """Return True when every trace has zero maximum absolute error."""
    return all(value["max_abs"] == 0.0 for value in metrics.values())


def _interpret_persisted(graph: Any, spikes: torch.Tensor, target: str) -> Any:
    """Save ``graph`` to ``target``, reload it, and interpret it."""
    save_graph(graph, target)
    return NirInterpreter(load_graph(target)).run(spikes)


def _report(
    path: str, reference: Any, reloaded: Any, readout: Dict[str, float]
) -> Dict[str, Any]:
    """Return the JSON-able fidelity report for one round-trip."""
    spikes = _trace_metrics(reloaded.spikes, reference.spikes)
    membranes = _trace_metrics(reloaded.membranes, reference.membranes)
    identical = (
        _exact(spikes)
        and _exact(membranes)
        and readout["max_abs"] == 0.0
        and reloaded.steps == reference.steps
    )
    return {
        "path": path,
        "steps": reloaded.steps,
        "identical": bool(identical),
        "readout": readout,
        "spikes": spikes,
        "membranes": membranes,
    }


def roundtrip(
    spec: TopologySpec, module: Any, spikes: torch.Tensor,
    graph: Optional[Any] = None, path: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist and reload a graph and report fidelity to the module's export.

    The reference is the independent interpretation of ``to_nir(spec,
    module)``. ``graph`` overrides the graph persisted (a perturbed parameter
    then shows as drift); ``path`` overrides the file location.
    """
    exported = to_nir(spec, module)
    persisted = exported if graph is None else graph
    reference = NirInterpreter(exported).run(spikes)
    with _target_path(path) as target:
        reloaded = _interpret_persisted(persisted, spikes, target)
    readout = drift.compare(reloaded.readout, reference.readout)
    return _report(target, reference, reloaded, readout)
