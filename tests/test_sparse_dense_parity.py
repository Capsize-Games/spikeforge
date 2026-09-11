"""Sparse-vs-dense readout parity and the sparse op-count reduction."""

import copy
from typing import Tuple

import pytest
import torch

from spikeforge.simulator.runner import run
from spikeforge.topology.registry import build_topology
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule
from spikeforge_targets.event_runtime import (
    DEFAULT_TOLERANCE,
    compare,
    matches,
    sparse_run,
)
from spikeforge_targets.event_runtime.spike_view import synthetic_spikes


def _pair(
    name: str, steps: int = 3
) -> Tuple[TopologySpec, StageModule, torch.Tensor]:
    """Return a seeded topology, module, and low-density spike fixture."""
    torch.manual_seed(0)
    spec, module = build_topology(name)
    return spec, module, synthetic_spikes(spec, steps, 2, 0, 0.1)


@pytest.mark.parametrize("name", ["fc_small", "conv_net", "recurrent_net"])
def test_sparse_readout_matches_dense(name: str) -> None:
    """The event-driven path reproduces the dense readout within tolerance."""
    _spec, module, spikes = _pair(name)
    with torch.no_grad():
        sparse = sparse_run(module, spikes)
        dense = run(module, spikes)
    assert matches(sparse, dense, DEFAULT_TOLERANCE)
    assert compare(sparse, dense)["max_abs"] <= DEFAULT_TOLERANCE


def test_sparse_reduces_ops_on_sparse_input() -> None:
    """A low-density fixture yields a strict SOP reduction over the MACs."""
    torch.manual_seed(0)
    spec, module = build_topology("conv_net")
    spikes = synthetic_spikes(spec, 4, 2, 1, 0.02)
    with torch.no_grad():
        sparse = sparse_run(module, spikes)
    block = compare(sparse, sparse)
    assert sparse.sop < sparse.mac
    assert sparse.sop / sparse.mac < 0.5
    assert block["efficiency"]["sop_over_mac"] == sparse.sop / sparse.mac


def test_dense_path_is_unchanged_by_a_sparse_run() -> None:
    """The dense default is untouched: same logits, same weights."""
    _spec, module, spikes = _pair("fc_small")
    before = copy.deepcopy(module.state_dict())
    with torch.no_grad():
        dense_first = run(module, spikes)
        sparse_run(module, spikes)
        dense_second = run(module, spikes)
    after = module.state_dict()
    assert torch.equal(dense_first.logits, dense_second.logits)
    for key, value in before.items():
        assert torch.equal(value, after[key])
