"""Tests for the headless verify CLI building blocks."""

import json
from typing import Tuple

import pytest
import torch

from spikeforge.cli import verify
from spikeforge.nir_bridge import to_nir
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule

pytest.importorskip("nir")

_Case = Tuple[TopologySpec, StageModule, torch.Tensor]


def _conv_case() -> _Case:
    """Return a small conv spec, module, and spatial spike train."""
    spec = presets.conv_net(
        in_channels=1, channels=2, num_classes=3, input_size=8
    )
    torch.manual_seed(0)
    return spec, build_module(spec), torch.rand(4, 2, 1, 8, 8)


def test_export_summary_is_jsonable() -> None:
    """``export_summary`` returns a JSON-able node/edge description."""
    summary = verify.export_summary("conv_net")
    assert summary["nodes"] and summary["edges"]
    assert json.loads(json.dumps(summary))["nodes"] == summary["nodes"]


def test_validate_report_passes_and_exits_zero() -> None:
    """A faithful module validates and maps to a zero exit status."""
    spec, module, spikes = _conv_case()
    report = verify.validate_report(spec, module, spikes)
    assert report["within_tolerance"] is True
    assert verify.exit_code(report) == 0
    assert json.dumps(report)


def test_perturbed_graph_exits_nonzero() -> None:
    """A perturbed graph flips the verdict and the exit status."""
    spec, module, spikes = _conv_case()
    graph = to_nir(spec, module)
    graph.nodes["conv1"].weight = graph.nodes["conv1"].weight + 0.5
    report = verify.validate_report(spec, module, spikes, graph=graph)
    assert report["within_tolerance"] is False
    assert verify.exit_code(report) == 1


def test_main_export_prints_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main`` dispatches ``export`` and returns success."""
    assert verify.main(["export", "--topology", "conv_net"]) == 0
    assert '"nodes"' in capsys.readouterr().out
