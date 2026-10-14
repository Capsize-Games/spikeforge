"""An embedding stage exports as a one-hot ingest plus an Affine node."""

import pytest
import torch
import torch.nn.functional as functional

from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.nir_bridge.mapper import map_stage
from spikeforge.nir_bridge.one_hot import one_hot
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import chain
from spikeforge.topology.stage import Stage

pytest.importorskip("nir")

#: Vocabulary and width shared by every fixture below.
_VOCAB = 5
_WIDTH = 4


def _stage() -> Stage:
    """Return the embedding stage every test exports."""
    return Stage(
        "emb", "embedding",
        {"num_embeddings": _VOCAB, "embedding_dim": _WIDTH},
    )


def test_embedding_maps_to_an_affine_node() -> None:
    """``map_stage`` renders embedding as a single ``Affine`` node."""
    mapping = map_stage(_stage())
    assert len(mapping.nodes) == 1
    assert type(mapping.nodes[0].node).__name__ == "Affine"


def test_embedding_copies_the_transposed_table() -> None:
    """The Affine weight is the table transpose with a zero bias."""
    torch.manual_seed(0)
    spec = chain([_stage()])
    module = build_module(spec)
    node = to_nir(spec, module).nodes["emb"]
    table = module.get_submodule("emb").weight.detach()
    assert type(node).__name__ == "Affine"
    assert tuple(node.weight.shape) == (_WIDTH, _VOCAB)
    assert torch.equal(torch.as_tensor(node.weight), table.T)
    assert torch.equal(
        torch.as_tensor(node.bias), torch.zeros(_WIDTH)
    )


def test_embedding_fidelity_is_float_exact() -> None:
    """One-hot products select rows bit-exactly on every-id fixtures."""
    torch.manual_seed(0)
    spec = chain([_stage()])
    module = build_module(spec)
    table = module.get_submodule("emb")
    node = to_nir(spec, module).nodes["emb"]
    weight = torch.as_tensor(node.weight)
    bias = torch.as_tensor(node.bias)
    fixtures = [
        torch.arange(_VOCAB).unsqueeze(0),
        torch.tensor([[3, 3, 3], [0, 4, 0]]),
        torch.tensor([[2]]),
        torch.tensor([[4], [0], [1], [2]]),
    ]
    for ids in fixtures:
        assert torch.equal(
            functional.linear(one_hot(ids, _VOCAB), weight, bias),
            table(ids),
        )


def test_embedding_graph_runs_through_the_interpreter() -> None:
    """The exported graph selects rows over a one-hot ``[T, B, L, V]``."""
    torch.manual_seed(0)
    spec = chain([_stage()])
    module = build_module(spec)
    ids = torch.tensor([[0, 4, 2], [1, 1, 3]])
    frames = one_hot(ids, _VOCAB).unsqueeze(0)
    result = NirInterpreter(to_nir(spec, module)).run(frames)
    assert torch.equal(result.readout, module.get_submodule("emb")(ids))


def test_structural_export_uses_a_zero_table() -> None:
    """No module means a zero ``[E, V]`` table with a zero bias."""
    node = to_nir(chain([_stage()])).nodes["emb"]
    assert type(node).__name__ == "Affine"
    assert torch.equal(
        torch.as_tensor(node.weight), torch.zeros(_WIDTH, _VOCAB)
    )
    assert torch.equal(
        torch.as_tensor(node.bias), torch.zeros(_WIDTH)
    )


def test_one_hot_layout_is_float_b_l_v() -> None:
    """Ingest maps long ``[B, L]`` ids to float32 ``[B, L, V]``."""
    hot = one_hot(torch.tensor([[0, 4], [2, 2]]), _VOCAB)
    assert hot.dtype == torch.float32
    assert tuple(hot.shape) == (2, 2, _VOCAB)
    expected = torch.zeros(2, 2, _VOCAB)
    expected[0, 0, 0] = 1.0
    expected[0, 1, 4] = 1.0
    expected[1, 0, 2] = 1.0
    expected[1, 1, 2] = 1.0
    assert torch.equal(hot, expected)


def test_one_hot_refuses_bad_frames() -> None:
    """Non-long, non-2D, out-of-range, and empty-vocab frames fail."""
    with pytest.raises(ValueError, match="long"):
        one_hot(torch.zeros(2, 3), _VOCAB)
    with pytest.raises(ValueError, match=r"\[B, L\]"):
        one_hot(torch.zeros(4, dtype=torch.long), _VOCAB)
    with pytest.raises(ValueError, match=r"\[0, num_embeddings\)"):
        one_hot(torch.tensor([[0, 5]]), _VOCAB)
    with pytest.raises(ValueError, match=r"\[0, num_embeddings\)"):
        one_hot(torch.tensor([[-1, 0]]), _VOCAB)
    with pytest.raises(ValueError, match="positive"):
        one_hot(torch.zeros(1, 1, dtype=torch.long), 0)
