"""C packages refuse graphs the C runtime cannot load, naming them."""

import os
from typing import Any

import pytest
import torch

from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.errors import CExportError
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import chain
from spikeforge.topology.stage import Stage

pytest.importorskip("nir")


def _linear_bundle() -> DeploymentBundle:
    """Return an in-memory bundle whose graph holds ``Linear`` nodes."""
    torch.manual_seed(0)
    spec = chain(
        [
            Stage(
                "fc1",
                "linear",
                {"in_features": 4, "out_features": 3, "bias": False},
            )
        ]
    )
    module = build_module(spec)
    return DeploymentBundle(
        manifest={"spec": spec.to_dict()},
        weights=dict(module.state_dict()),
    )


def _mixed_bundle() -> DeploymentBundle:
    """Return a bundle whose graph holds ``Conv1d`` and ``Linear``."""
    torch.manual_seed(0)
    spec = chain(
        [
            Stage(
                "c1",
                "conv1d",
                {
                    "in_channels": 2,
                    "out_channels": 3,
                    "kernel_size": 3,
                },
            ),
            Stage(
                "fc1",
                "linear",
                {"in_features": 4, "out_features": 3, "bias": False},
            ),
        ]
    )
    module = build_module(spec)
    return DeploymentBundle(
        manifest={"spec": spec.to_dict()},
        weights=dict(module.state_dict()),
    )


def test_unsupported_kind_is_refused_with_its_name(tmp_path: Any) -> None:
    """A ``Linear`` graph is refused by a typed error naming the kind."""
    with pytest.raises(CExportError) as caught:
        _linear_bundle().export_c_package(str(tmp_path / "c_pkg"))
    assert caught.value.kinds == ("Linear",)
    assert "'Linear'" in str(caught.value)


def test_unsupported_kinds_are_all_named(tmp_path: Any) -> None:
    """Every rejected kind is named, and no package is written."""
    target = str(tmp_path / "c_pkg")
    with pytest.raises(CExportError) as caught:
        _mixed_bundle().export_c_package(target)
    assert caught.value.kinds == ("Conv1d", "Linear")
    assert not os.path.exists(target)
