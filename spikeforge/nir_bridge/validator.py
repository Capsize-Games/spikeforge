"""Compare a snnTorch topology module against its exported NIR graph.

The snnTorch module is run through the shared simulator and the exported
graph through the independent :class:`NirInterpreter`; the two trajectories
are then compared per neuron stage and at the readout. Spikes are expected
to match exactly and numeric quantities within a small epsilon. The result
is a JSON-serialisable :class:`ValidationReport`.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from spikeforge.neurons.registry import NEURONS
from spikeforge.nir_bridge import drift, node_names
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.nir_bridge.tolerances import resolved
from spikeforge.nir_bridge.validation_report import ValidationReport
from spikeforge.observability import metrics as obs_metrics
from spikeforge.simulator.runner import run
from spikeforge.topology.spec import TopologySpec

#: Explains why membranes are compared directly, without rescaling.
SHARED_UNITS_NOTE = (
    "membrane compared in native units: R = 1/(1-beta) makes the "
    "zero-order-hold input gain exactly one, so both sides share units"
)

_Entry = Tuple[str, str, Mapping[str, float]]


def quantity_key(quantity: str) -> str:
    """Return the tolerance prefix for a layer quantity name."""
    return "spike" if quantity == "spikes" else quantity


def within(
    quantity: str, metrics: Mapping[str, float], settings: Mapping[str, float]
) -> bool:
    """Return True when ``metrics`` stay inside ``settings``."""
    prefix = quantity_key(quantity)
    if metrics["max_abs"] > settings[f"{prefix}_max_abs"]:
        return False
    if metrics["mean_abs"] > settings[f"{prefix}_mean_abs"]:
        return False
    if prefix == "spike":
        return metrics["agreement"] >= settings["spike_agreement"]
    return True


def _compare_layer(
    name: str, kind: str, reference: Any, result: Any
) -> Dict[str, Any]:
    """Return the metric mapping for one neuron stage."""
    layer: Dict[str, Any] = {}
    spike = node_names.spike_node(name)
    if spike in result.spikes and name in reference.spikes:
        layer["spikes"] = drift.compare(
            result.spikes[spike], reference.spikes[name]
        )
    membrane = node_names.find_membrane(name, kind, result.membranes)
    if membrane is not None and name in reference.membranes:
        layer["membrane"] = drift.compare(
            result.membranes[membrane], reference.membranes[name]
        )
    return layer


def _quantities(layer: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    """Return the metric entries of a layer, skipping its pass flag."""
    return [
        (quantity, metrics)
        for quantity, metrics in layer.items()
        if isinstance(metrics, dict)
    ]


def _layers(
    spec: TopologySpec,
    reference: Any,
    result: Any,
    settings: Mapping[str, float],
) -> Dict[str, Any]:
    """Return the per-stage metric mapping of a validation run."""
    layers: Dict[str, Any] = {}
    for stage in spec.stages:
        if stage.kind not in NEURONS:
            continue
        layer = _compare_layer(stage.name, stage.kind, reference, result)
        layer["within_tolerance"] = bool(layer) and all(
            within(quantity, metrics, settings)
            for quantity, metrics in _quantities(layer)
        )
        layers[stage.name] = layer
    return layers


def _failures(
    layers: Mapping[str, Any], settings: Mapping[str, float]
) -> List[_Entry]:
    """Return every ``(layer, quantity, metrics)`` that breached tolerance."""
    found: List[_Entry] = []
    for name, layer in layers.items():
        for quantity, metrics in _quantities(layer):
            if not within(quantity, metrics, settings):
                found.append((name, quantity, metrics))
    return found


def _metric(
    quantity: str, metrics: Mapping[str, float], settings: Mapping[str, float]
) -> str:
    """Return the metric name that best explains a breach."""
    prefix = quantity_key(quantity)
    disagreed = metrics["agreement"] < settings["spike_agreement"]
    if prefix == "spike" and disagreed:
        return "agreement"
    if metrics["max_abs"] > settings[f"{prefix}_max_abs"]:
        return "max_abs"
    return "mean_abs"


def _descriptor(
    entry: _Entry, settings: Mapping[str, float]
) -> Dict[str, Any]:
    """Return a JSON-able descriptor naming a layer, quantity and metric."""
    name, quantity, metrics = entry
    metric = _metric(quantity, metrics, settings)
    return {
        "layer": name,
        "quantity": quantity,
        "metric": metric,
        "value": metrics[metric],
    }


def _largest(
    layers: Mapping[str, Any], readout: Mapping[str, float]
) -> _Entry:
    """Return the entry with the greatest max absolute error."""
    best: _Entry = ("readout", "readout", readout)
    for name, layer in layers.items():
        for quantity, metrics in _quantities(layer):
            if metrics["max_abs"] > best[2]["max_abs"]:
                best = (name, quantity, metrics)
    return best


def _worst(
    layers: Mapping[str, Any],
    readout: Mapping[str, float],
    settings: Mapping[str, float],
) -> Dict[str, Any]:
    """Return the worst descriptor, preferring the first root-cause breach."""
    failures = _failures(layers, settings)
    if failures:
        return _descriptor(failures[0], settings)
    if not within("readout", readout, settings):
        return _descriptor(("readout", "readout", readout), settings)
    return _descriptor(_largest(layers, readout), settings)


def _trajectories(
    spec: TopologySpec, module: Any, spikes: torch.Tensor, graph: Optional[Any]
) -> Tuple[Any, Any]:
    """Return the snnTorch trajectory and the NIR result for ``spikes``."""
    reference = run(module, spikes, track=True, membrane=True)
    exported = graph if graph is not None else to_nir(spec, module)
    return reference, NirInterpreter(exported).run(spikes)


def _verdict(
    layers: Mapping[str, Any],
    readout: Mapping[str, float],
    settings: Mapping[str, float],
) -> Tuple[bool, Dict[str, Any]]:
    """Return the overall pass flag and the worst descriptor."""
    passed = not _failures(layers, settings) and within(
        "readout", readout, settings
    )
    return passed, _worst(layers, readout, settings)


def validate(
    spec: TopologySpec,
    module: Any,
    spikes: torch.Tensor,
    tolerances: Optional[Mapping[str, float]] = None,
    graph: Optional[Any] = None,
) -> ValidationReport:
    """Compare ``module`` with its exported NIR graph over ``spikes``.

    ``tolerances`` overrides the defaults; ``graph`` supplies an already
    exported graph instead of re-exporting ``module``.
    """
    settings = resolved(tolerances)
    reference, result = _trajectories(spec, module, spikes, graph)
    layers = _layers(spec, reference, result, settings)
    readout = drift.compare(result.readout, reference.logits)
    obs_metrics.gauge("validation.drift", float(readout["max_abs"]))
    obs_metrics.counter("validation.nir_runs")
    passed, worst = _verdict(layers, readout, settings)
    return ValidationReport(
        passed, reference.steps, layers, readout, worst, [SHARED_UNITS_NOTE]
    )
