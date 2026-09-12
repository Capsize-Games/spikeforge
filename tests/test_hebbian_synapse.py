"""Tests for the one-shot Hebbian synapse (issue #22)."""

import torch

from spikeforge.memory.hebbian_synapse import HebbianSynapse


def test_starts_with_no_output_columns() -> None:
    """A fresh synapse has no taught classes yet."""
    synapse = HebbianSynapse(in_features=4)
    assert synapse.out_features == 0


def test_grow_appends_a_zero_column() -> None:
    """Growing adds one all-zero column and returns its index."""
    synapse = HebbianSynapse(in_features=4)
    index = synapse.grow()
    assert index == 0
    assert synapse.out_features == 1
    spikes = torch.ones(1, 4)
    assert torch.equal(synapse.forward(spikes), torch.zeros(1, 1))


def test_write_only_changes_its_own_column() -> None:
    """Writing to one column leaves every other column untouched."""
    synapse = HebbianSynapse(in_features=3)
    first = synapse.grow()
    second = synapse.grow()
    synapse.write(torch.tensor([1.0, 0.0, 0.0]), first, gain=2.0)
    assert synapse.forward(torch.eye(3))[:, second].abs().sum() == 0.0


def test_write_is_normalised_before_scaling() -> None:
    """A write's strength depends on ``gain``, not the trace's scale."""
    small = HebbianSynapse(in_features=2)
    large = HebbianSynapse(in_features=2)
    small.write(torch.tensor([1.0, 0.0]), small.grow(), gain=3.0)
    large.write(torch.tensor([10.0, 0.0]), large.grow(), gain=3.0)
    probe = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(small.forward(probe), large.forward(probe))
