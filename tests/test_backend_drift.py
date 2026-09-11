"""Backend trajectory comparison against the reference interpreter."""

import json
from dataclasses import replace
from typing import Any

import pytest
import torch

from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import chain
from spikeforge.topology.stage import Stage
from spikeforge_targets.backends import compile_run
from spikeforge_targets.backends.compare import compare_results
from spikeforge_targets.backends.reference_backend import (
    ReferenceBackend,
)

pytest.importorskip("nir")


def _graph() -> Any:
    """Return a small zero-reset chain graph."""
    spec = chain([
        Stage("fc1", "linear", {"in_features": 6, "out_features": 4}),
        Stage("lif1", "leaky", {"beta": 0.8, "reset": "zero"}),
        Stage("out", "linear", {"in_features": 4, "out_features": 2}),
    ])
    return to_nir(spec, build_module(spec))


def _spikes() -> torch.Tensor:
    """Return a fixed-seed spike train."""
    torch.manual_seed(0)
    return torch.rand(5, 2, 6)


def test_reference_backend_matches_the_interpreter() -> None:
    """The reference backend executes exactly the reference interpreter."""
    graph, spikes = _graph(), _spikes()
    backend = ReferenceBackend()
    result = backend.run(backend.compile(graph, None), spikes)
    expected = NirInterpreter(graph).run(spikes)
    assert torch.allclose(result.readout, expected.readout)
    assert sorted(result.spikes) == sorted(expected.spikes)


def test_compare_is_zero_for_identical_trajectories() -> None:
    """A backend result compared to the reference shows no drift."""
    graph, spikes = _graph(), _spikes()
    run = compile_run("reference", graph, spikes)
    reference = ReferenceBackend().run(graph, spikes)
    compare = compare_results(run, reference)
    assert compare["readout"]["max_abs"] == 0.0
    assert compare["spikes"]["agreement"] == 1.0


def test_compare_detects_a_perturbed_readout() -> None:
    """A perturbed readout produces a nonzero drift metric."""
    graph, spikes = _graph(), _spikes()
    reference = ReferenceBackend().run(graph, spikes)
    perturbed = replace(reference, readout=reference.readout + 1.0)
    compare = compare_results(perturbed, reference)
    assert compare["readout"]["max_abs"] == pytest.approx(1.0)


def test_backend_result_dict_is_json_serialisable() -> None:
    """The run result serialises with its readout and node names."""
    result = compile_run("reference", _graph(), _spikes())
    payload = result.to_dict()
    assert json.dumps(payload)
    assert payload["status"] == "ok"
    assert payload["steps"] == 5
    assert payload["readout"]
    assert payload["compare"]["readout"]["max_abs"] == 0.0
