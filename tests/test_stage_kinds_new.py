"""Every new module kind builds a torch module with the expected shape."""

from typing import Any, Dict, Tuple

import pytest
import torch
import torch.nn as nn

from spikeforge.nir_bridge import stage_builders
from spikeforge.nir_bridge.errors import UnsupportedStageError
from spikeforge.nir_bridge.mapper import map_stage
from spikeforge.topology import stage_modules
from spikeforge.topology.stage import Stage

pytest.importorskip("nir")

_Shapes = Tuple[int, ...]
_Case = Tuple[str, Dict[str, Any], _Shapes, _Shapes]

_CONV1D: Dict[str, Any] = {
    "in_channels": 2, "out_channels": 3, "kernel_size": 3,
}

_FLOAT_CASES: Tuple[_Case, ...] = (
    ("conv1d", _CONV1D, (2, 2, 10), (2, 3, 8)),
    ("maxpool1d", {"kernel_size": 2}, (2, 3, 8), (2, 3, 4)),
    ("maxpool2d", {"kernel_size": 2}, (2, 3, 8, 8), (2, 3, 4, 4)),
    ("layer_norm", {"normalized_shape": 4}, (2, 5, 4), (2, 5, 4)),
    ("batch_norm", {"num_features": 4}, (2, 4, 5), (2, 4, 5)),
    ("dropout", {"p": 0.5}, (2, 5, 4), (2, 5, 4)),
    ("positional_encoding", {"embed_dim": 4}, (2, 5, 4), (2, 5, 4)),
    ("attention", {"embed_dim": 4}, (2, 5, 4), (2, 5, 4)),
    (
        "multihead_attention",
        {"embed_dim": 4, "num_heads": 2},
        (2, 5, 4), (2, 5, 4),
    ),
)


@pytest.mark.parametrize("kind,params,in_shape,out_shape", _FLOAT_CASES)
def test_new_kind_builds_and_forwards(
    kind: str, params: Dict[str, Any], in_shape: _Shapes, out_shape: _Shapes
) -> None:
    """Each float kind builds a module and maps its input shape as expected."""
    module = stage_modules.module_for_stage(Stage(kind, kind, params))
    assert isinstance(module, nn.Module)
    module.eval()
    assert module(torch.randn(*in_shape)).shape == out_shape


def test_embedding_looks_up_token_vectors() -> None:
    """``embedding`` maps integer tokens to their vectors."""
    params = {"num_embeddings": 5, "embedding_dim": 4}
    module = stage_modules.module_for_stage(Stage("e", "embedding", params))
    assert module(torch.randint(0, 5, (2, 6))).shape == (2, 6, 4)


def test_dropout_is_the_identity_at_inference() -> None:
    """``dropout`` returns its input unchanged in eval mode."""
    module = stage_modules.module_for_stage(
        Stage("d", "dropout", {"p": 0.5})
    )
    module.eval()
    x = torch.randn(2, 7)
    assert torch.equal(module(x), x)


def test_conv1d_contract_is_mapped_when_primitive_present() -> None:
    """The live probe reports mapped for the installed ``nir.Conv1d``."""
    assert stage_builders.conv1d_contract() == stage_builders.MAPPED


def test_conv1d_contract_reports_unexportable_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe flips to unexportable when the primitive is missing."""
    monkeypatch.setattr(stage_builders.api, "node_class", lambda name: None)
    assert stage_builders.conv1d_contract() == stage_builders.UNEXPORTABLE


def test_conv1d_mapping_fails_when_primitive_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing primitive raises the typed error naming the stage."""
    monkeypatch.setattr(stage_builders.api, "node_class", lambda name: None)
    with pytest.raises(UnsupportedStageError) as excinfo:
        map_stage(Stage("c", "conv1d", _CONV1D))
    assert excinfo.value.kind == "conv1d"


def test_conv1d_maps_to_a_conv1d_node() -> None:
    """A conv1d stage maps to a ``nir.Conv1d`` node."""
    node = map_stage(Stage("c", "conv1d", _CONV1D)).nodes[0].node
    assert type(node).__name__ == "Conv1d"


def test_dropout_is_a_documented_passthrough() -> None:
    """``dropout`` emits no node and resolves to its neighbours."""
    mapping = map_stage(Stage("d", "dropout", {"p": 0.25}))
    assert mapping.nodes == ()
    assert mapping.entry is None
    assert mapping.output is None
