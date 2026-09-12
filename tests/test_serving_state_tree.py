"""The serializable carried state of a streaming inference session."""

import torch

from spikeforge.serving.state_tree import StateTree


def test_round_trip_preserves_tensors_and_dtypes() -> None:
    """A nested state survives encode/decode with dtype and shape intact."""
    state = {
        "lif1": (torch.rand(2, 3),),
        "__prev__": {"fc1": torch.zeros(2, 4)},
    }
    restored = StateTree.from_dict(StateTree(state).to_dict()).raw()
    assert torch.equal(restored["lif1"][0], state["lif1"][0])
    assert restored["lif1"][0].dtype == state["lif1"][0].dtype
    assert torch.equal(restored["__prev__"]["fc1"], state["__prev__"]["fc1"])


def test_rebuilds_neuron_tuples_not_lists() -> None:
    """A neuron state stays an ordered tuple after a round trip."""
    payload = StateTree({"syn": (torch.zeros(1), torch.zeros(1))}).to_dict()
    restored = StateTree.from_dict(payload).raw()
    assert isinstance(restored["syn"], tuple)
    assert len(restored["syn"]) == 2


def test_reset_clears_every_carried_tensor() -> None:
    """Reset drops the whole carried state."""
    tree = StateTree({"lif1": (torch.ones(1),)})
    tree.reset()
    assert tree.raw() == {}
    assert tree.to_dict() == {}


def test_handles_empty_and_none_leaves() -> None:
    """An unmaterialised neuron state and a None leaf both round-trip."""
    payload = StateTree({"lif1": (), "lif2": None}).to_dict()
    restored = StateTree.from_dict(payload).raw()
    assert restored["lif1"] == ()
    assert restored["lif2"] is None


def test_to_places_every_tensor_on_the_requested_device() -> None:
    """Device moves reach tensors nested in tuples and dicts."""
    tree = StateTree(
        {"lif1": (torch.zeros(1),), "__current__": {"lif1": torch.ones(1)}}
    )
    moved = tree.to("cpu")
    assert moved.raw()["lif1"][0].device.type == "cpu"
    assert moved.raw()["__current__"]["lif1"].device.type == "cpu"
