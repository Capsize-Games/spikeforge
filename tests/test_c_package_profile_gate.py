"""The C package gate consumes the single-sourced C runtime profile."""

import os
from typing import Any, Callable, Dict, Tuple

import pytest
import torch

from spikeforge.nir_bridge import exporter
from spikeforge.serving import c_package
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.c_package import GRAPH_FILENAME
from spikeforge.serving.errors import CExportError
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import chain
from spikeforge.topology.stage import Stage
from spikeforge_targets import c_runtime

pytest.importorskip("nir")

#: A summary mutation standing in for one unloadable graph shape.
_Doctor = Callable[[Dict[str, Any]], None]


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
    """Return a bundle whose graph holds Conv2d, Flatten, and Affine."""
    torch.manual_seed(0)
    spec = chain(
        [
            Stage(
                "conv",
                "conv2d",
                {
                    "in_channels": 2,
                    "out_channels": 3,
                    "kernel_size": 3,
                },
            ),
            Stage("flat", "flatten", {}),
            Stage("fc", "linear", {"in_features": 6, "out_features": 2}),
        ]
    )
    module = build_module(spec)
    return DeploymentBundle(
        manifest={"spec": spec.to_dict()},
        weights=dict(module.state_dict()),
    )


def _drop_affine_bias(summary: Dict[str, Any]) -> None:
    """Delete the Affine bias so the C shape check fires."""
    for record in summary["nodes"]:
        if record["kind"] == "Affine":
            del record["params"]["bias"]


def _flatten_partially(summary: Dict[str, Any]) -> None:
    """Narrow the Flatten dims so the C all-dims check fires."""
    for record in summary["nodes"]:
        if record["kind"] == "Flatten":
            record["params"]["start_dim"] = 0


def _scalarize_stride(summary: Dict[str, Any]) -> None:
    """Collapse the Conv2d stride so the C grid check fires."""
    for record in summary["nodes"]:
        if record["kind"] == "Conv2d":
            record["params"]["stride"] = 1


def _close_cycle(summary: Dict[str, Any]) -> None:
    """Loop an edge back so the C unbroken-cycle check fires."""
    summary["edges"].append(
        {"source": "fc", "target": "fc", "delayed": False}
    )


#: Doctor, node, kind, and reason fragment of each unloadable shape.
_CASES: Tuple[Tuple[_Doctor, str, str, str], ...] = (
    (_drop_affine_bias, "fc", "Affine", "bias"),
    (_flatten_partially, "flat", "Flatten", "start_dim"),
    (_scalarize_stride, "conv", "Conv2d", "stride"),
    (_close_cycle, "graph", "graph", "Delay"),
)


def test_gate_consumes_the_single_sourced_profile() -> None:
    """The export allow-list IS the profile's kind set, not a copy."""
    assert c_package.SUPPORTED_NODE_KINDS is c_runtime.SUPPORTED


def test_supported_graph_still_exports(tmp_path: Any) -> None:
    """A conforming mixed graph writes its package without complaint."""
    package = _mixed_bundle().export_c_package(str(tmp_path / "c_pkg"))
    assert os.path.isfile(os.path.join(package, GRAPH_FILENAME))


def test_kind_refusal_names_nodes_and_kinds(tmp_path: Any) -> None:
    """A ``Linear`` graph refuses with per-node names, kinds, reasons."""
    target = str(tmp_path / "c_pkg")
    with pytest.raises(CExportError) as caught:
        _linear_bundle().export_c_package(target)
    assert caught.value.kinds == ("Linear",)
    assert len(caught.value.violations) == 1
    name, kind, reason = caught.value.violations[0]
    assert (name, kind) == ("fc1", "Linear")
    assert "Linear" in reason
    assert "'fc1'" in str(caught.value)
    assert not os.path.exists(target)


@pytest.mark.parametrize("doctor,node,kind,fragment", _CASES)
def test_profile_shape_violations_refuse_early(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    doctor: _Doctor,
    node: str,
    kind: str,
    fragment: str,
) -> None:
    """Shape violations refuse naming the node; no package is written."""
    real = exporter.graph_summary

    def _doctored(target: Any) -> Dict[str, Any]:
        """Return the real summary with one unloadable shape."""
        summary = real(target)
        doctor(summary)
        return summary

    monkeypatch.setattr(exporter, "graph_summary", _doctored)
    target = str(tmp_path / "c_pkg")
    with pytest.raises(CExportError) as caught:
        _mixed_bundle().export_c_package(target)
    assert caught.value.kinds == (kind,)
    assert len(caught.value.violations) == 1
    found = caught.value.violations[0]
    assert (found[0], found[1]) == (node, kind)
    assert fragment in found[2]
    assert f"'{node}'" in str(caught.value)
    assert not os.path.exists(target)
