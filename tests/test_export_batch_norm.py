"""A 2D eval batch norm stage exports as an Affine fold."""

import math
from typing import Any, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as functional

from spikeforge.nir_bridge.affine_folds import batch_norm_node
from spikeforge.nir_bridge.errors import UnsupportedStageError
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec, chain
from spikeforge.topology.stage import Stage
from spikeforge.topology.stage_module import StageModule

pytest.importorskip("nir")

#: Channel count shared by every fixture below.
_FEATURES = 4


def _loaded() -> Tuple[TopologySpec, StageModule]:
    """Return a spec and eval module with non-trivial frozen buffers."""
    spec = chain(
        [Stage("bn", "batch_norm", {"num_features": _FEATURES})]
    )
    module = build_module(spec)
    norm = module.get_submodule("bn")
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, -1.5, 2.0, 0.25]))
        norm.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
        norm.running_mean.copy_(torch.tensor([1.0, -1.0, 0.5, 2.0]))
        norm.running_var.copy_(torch.tensor([0.25, 4.0, 1.0, 0.5]))
    module.eval()
    return spec, module


def _folded(norm: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(weight, bias)`` fold the issue constructs."""
    var = norm.running_var.detach()
    scale = norm.weight.detach() / torch.sqrt(var + norm.eps)
    weight = torch.diag(scale)
    bias = norm.bias.detach() - norm.running_mean.detach() * scale
    return weight, bias


def test_batch_norm_folds_into_a_diagonal_affine() -> None:
    """The node carries the issue's folded weight and bias exactly."""
    spec, module = _loaded()
    node = to_nir(spec, module).nodes["bn"]
    assert type(node).__name__ == "Affine"
    assert tuple(node.weight.shape) == (_FEATURES, _FEATURES)
    weight, bias = _folded(module.get_submodule("bn"))
    assert torch.equal(torch.as_tensor(node.weight), weight)
    assert torch.equal(torch.as_tensor(node.bias), bias)


def test_batch_norm_fidelity_within_fold_rounding() -> None:
    """The fold matches eval batch norm on ``[B, C]`` fixtures."""
    spec, module = _loaded()
    norm = module.get_submodule("bn")
    node = to_nir(spec, module).nodes["bn"]
    weight = torch.as_tensor(node.weight)
    bias = torch.as_tensor(node.bias)
    torch.manual_seed(1)
    fixtures = [
        torch.randn(1, _FEATURES) * 3.0,
        torch.randn(5, _FEATURES),
        torch.randn(2, _FEATURES) - 5.0,
    ]
    for frame in fixtures:
        want = norm(frame).detach()
        got = functional.linear(frame, weight, bias)
        assert float((want - got).abs().max()) <= 1e-5


def test_batch_norm_without_affine_folds_identity() -> None:
    """An ``affine=False`` module folds gamma=1, beta=0 faithfully."""
    norm = nn.BatchNorm1d(_FEATURES, affine=False)
    norm.running_mean.copy_(torch.tensor([0.5, 0.0, -2.0, 1.5]))
    norm.running_var.copy_(torch.tensor([2.0, 0.5, 1.0, 3.0]))
    norm.eval()
    node = batch_norm_node(
        "bn", {"num_features": _FEATURES}, norm
    ).node
    scale = 1.0 / torch.sqrt(norm.running_var + norm.eps)
    assert torch.equal(torch.as_tensor(node.weight), torch.diag(scale))
    torch.manual_seed(2)
    frame = torch.randn(3, _FEATURES)
    want = norm(frame).detach()
    got = functional.linear(
        frame, torch.as_tensor(node.weight), torch.as_tensor(node.bias)
    )
    assert float((want - got).abs().max()) <= 1e-5


def test_structural_export_folds_default_buffers() -> None:
    """No module means the fresh-module fold: scaled identity, zero bias."""
    params = {"num_features": _FEATURES}
    node = to_nir(chain([Stage("bn", "batch_norm", params)])).nodes["bn"]
    assert type(node).__name__ == "Affine"
    fresh = nn.BatchNorm1d(_FEATURES)
    fresh.eval()
    reference = batch_norm_node("bn", params, fresh).node
    assert torch.equal(
        torch.as_tensor(node.weight), torch.as_tensor(reference.weight)
    )
    assert torch.equal(
        torch.as_tensor(node.bias), torch.zeros(_FEATURES)
    )
    assert float(node.weight[0][0]) == pytest.approx(
        1.0 / math.sqrt(1.0 + 1e-5)
    )


def test_training_batch_norm_stays_simulation_only() -> None:
    """A training-mode module raises naming batch statistics, no record."""
    spec = chain(
        [Stage("bn", "batch_norm", {"num_features": _FEATURES})]
    )
    module = build_module(spec)
    assert module.get_submodule("bn").training
    with pytest.raises(UnsupportedStageError) as caught:
        to_nir(spec, module)
    assert caught.value.kind == "batch_norm"
    assert "batch statistics" in str(caught.value)
    assert caught.value.hint == ""


def test_batch_norm2d_stays_simulation_only() -> None:
    """``dim=2`` refuses with and without a module."""
    spec = chain(
        [Stage("bn", "batch_norm", {"num_features": 3, "dim": 2})]
    )
    with pytest.raises(UnsupportedStageError) as caught:
        to_nir(spec)
    assert caught.value.kind == "batch_norm"
    with pytest.raises(UnsupportedStageError) as caught:
        to_nir(spec, build_module(spec))
    assert caught.value.kind == "batch_norm"


def test_batch_norm2d_module_refused_by_class() -> None:
    """A BatchNorm2d submodule refuses even with default params."""
    with pytest.raises(UnsupportedStageError) as caught:
        batch_norm_node("bn", {"num_features": 3}, nn.BatchNorm2d(3))
    assert caught.value.kind == "batch_norm"
    assert "BatchNorm2d" in str(caught.value)


def test_batch_norm_without_stats_stays_simulation_only() -> None:
    """Stat-less modules use batch statistics even in eval mode."""
    norm = nn.BatchNorm1d(_FEATURES, track_running_stats=False)
    norm.eval()
    with pytest.raises(UnsupportedStageError) as caught:
        batch_norm_node("bn", {"num_features": _FEATURES}, norm)
    assert caught.value.kind == "batch_norm"
    assert "batch statistics" in str(caught.value)
