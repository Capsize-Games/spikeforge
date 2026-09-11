"""Phase 1 vertical slice: conv_net -> NIR -> independent interpreter.

This is the roadmap's recommended first end-to-end proof: build the
convolutional preset, export it to a NIR graph, execute that graph with the
independent reference interpreter, and show the validation report is within
tolerance with zero spike drift and a bit-exact readout. It needs no dataset
download: a randomly initialised module and a synthetic spike train suffice.
"""

from typing import Tuple

import pytest
import torch

from spikeforge.neurons.registry import NEURONS
from spikeforge.nir_bridge import node_names
from spikeforge.nir_bridge.drift import compare
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.nir_bridge.validator import validate
from spikeforge.simulator.runner import run
from spikeforge.topology.registry import build_topology
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule

pytest.importorskip("nir")

_Case = Tuple[TopologySpec, StageModule, torch.Tensor]


def _conv_case() -> _Case:
    """Build a small conv_net and a matching spatial spike train."""
    torch.manual_seed(0)
    spec, module = build_topology(
        "conv_net", {"channels": 2, "num_classes": 3, "input_size": 8}
    )
    return spec, module, torch.rand(5, 2, 1, 8, 8)


def test_slice_exports_and_validates_within_tolerance() -> None:
    """The exported conv_net graph validates within tolerance."""
    spec, module, spikes = _conv_case()
    report = validate(spec, module, spikes)
    assert report["within_tolerance"] is True
    assert report["layers"]


def test_slice_independent_interpreter_has_zero_spike_drift() -> None:
    """The independent interpreter matches every spike bit for bit."""
    spec, module, spikes = _conv_case()
    reference = run(module, spikes, track=True)
    result = NirInterpreter(to_nir(spec, module)).run(spikes)
    compared = 0
    for stage in spec.stages:
        if stage.kind not in NEURONS:
            continue
        node = node_names.spike_node(stage.name)
        if node not in result.spikes:
            continue
        metrics = compare(result.spikes[node], reference.spikes[stage.name])
        assert metrics["max_abs"] == 0.0
        assert metrics["agreement"] == 1.0
        compared += 1
    assert compared >= 1
    assert compare(result.readout, reference.logits)["max_abs"] == 0.0
