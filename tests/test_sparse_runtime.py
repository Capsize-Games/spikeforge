"""The sparse/event-driven runtime: counting and typed failures."""

import pytest
import torch

from spikeforge.topology.builder import build_module
from spikeforge.topology.registry import build_topology
from spikeforge.topology.spec import chain
from spikeforge.topology.stage import Stage
from spikeforge_targets.event_runtime import (
    SparseResult,
    UnsupportedKindError,
    sparse_run,
)
from spikeforge_targets.event_runtime.spike_view import SparseSpikes


def _known_graph() -> torch.nn.Module:
    """Return a 4->3 linear plus a leaky neuron with fixed input."""
    spec = chain([
        Stage("fc", "linear", {"in_features": 4, "out_features": 3}),
        Stage("lif", "leaky", {"beta": 0.5}),
    ])
    return build_module(spec)


def test_sparse_run_returns_a_sparse_result() -> None:
    """A sparse run yields a Trajectory plus counter and density helpers."""
    torch.manual_seed(0)
    spec, module = build_topology("fc_small")
    spikes = torch.zeros(2, 1, 28 * 28)
    spikes[0, 0, :10] = 1.0
    with torch.no_grad():
        result = sparse_run(module, spikes)
    assert isinstance(result, SparseResult)
    assert result.steps == 2
    assert result.logits.shape == torch.Size([1, 10])
    assert set(result.counts) == {"sop", "mac", "ac", "timesteps"}
    assert result.counts["timesteps"] == 2
    assert result.sop > 0 and result.mac > result.sop
    assert 0.0 < result.density < 1.0
    assert result.per_stage["fc1"]["mac"] > 0


def test_counts_known_small_graph() -> None:
    """Exact SOP/MAC/AC for a two-stage graph with two active inputs."""
    module = _known_graph()
    spikes = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    with torch.no_grad():
        result = sparse_run(module, spikes)
    assert result.counts == {"sop": 6, "mac": 12, "ac": 6, "timesteps": 1}
    assert result.per_stage["fc"] == {"sop": 6, "mac": 12, "ac": 3}
    assert result.per_stage["lif"] == {"sop": 0, "mac": 0, "ac": 3}
    assert result.density == pytest.approx(0.5)


def test_counters_off_omits_the_tally() -> None:
    """``counters=False`` still returns a trajectory but no counts."""
    module = _known_graph()
    spikes = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    with torch.no_grad():
        result = sparse_run(module, spikes, counters=False)
    assert result.counts == {}
    assert result.per_stage == {}
    assert result.steps == 1


def test_unsupported_kind_raises_typed_error() -> None:
    """A parameterised kind with no event-driven op raises, never degrades."""
    spec = chain([Stage("ln", "layer_norm", {"normalized_shape": 4})])
    module = build_module(spec)
    spikes = torch.rand(1, 1, 4)
    with pytest.raises(UnsupportedKindError) as excinfo:
        sparse_run(module, spikes)
    assert excinfo.value.kind == "layer_norm"
    assert excinfo.value.stage == "ln"


def test_sparse_spikes_density_helpers() -> None:
    """The COO view reports nnz, dense size, and density."""
    frame = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    view = SparseSpikes.from_dense(frame)
    assert view.nnz == 2
    assert view.numel == 6
    assert view.density == pytest.approx(2 / 6)
    assert view.torch_sparse().to_dense().equal(frame)
