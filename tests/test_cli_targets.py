"""Tests for the Phase 5c deployment CLI subcommands."""

import json
from pathlib import Path
from typing import Any

import pytest

from spikeforge.cli import verify
from spikeforge.nir_bridge import save_graph, to_nir
from spikeforge.topology.registry import build_topology
from spikeforge_targets import registry
from spikeforge_targets.cli import target_cli

pytest.importorskip("nir")


def test_targets_payload_lists_every_target() -> None:
    """The registry payload carries availability for every target."""
    payload = target_cli.targets_payload()
    names = [item["name"] for item in payload["targets"]]
    assert names == registry.target_names()
    for item in payload["targets"]:
        assert {"kind", "extra", "available", "supported_count"} <= set(item)
    assert json.dumps(payload)


def test_main_targets_prints_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``targets`` prints the registry as JSON and exits zero."""
    assert verify.main(["targets"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["targets"]


def test_deploy_report_reference_is_deployable() -> None:
    """The reference target deploys a preset and maps to a zero status."""
    report = target_cli.deploy_report("conv_net", "reference")
    assert json.dumps(report)
    assert report["deployable"] is True
    assert target_cli.deploy_exit(report) == 0


def test_deploy_unavailable_target_exits_nonzero() -> None:
    """A target without its SDK is undeployable and exits non-zero."""
    report = target_cli.deploy_report("conv_net", "norse")
    assert report["deployable"] is False
    assert target_cli.deploy_exit(report) == 1


def test_main_deploy_failure_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``deploy`` prints the report but returns 1 for a gap."""
    args = ["deploy", "--topology", "conv_net", "--target", "norse"]
    assert verify.main(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["deployable"] is False


def test_roundtrip_report_is_identical() -> None:
    """A fresh preset round-trips through disk without drift."""
    report = target_cli.roundtrip_report("conv_net")
    assert json.dumps(report)
    assert report["identical"] is True
    assert target_cli.roundtrip_exit(report) == 0


def _perturbed_graph() -> Any:
    """Return a conv graph whose first convolution weight was changed."""
    spec, module = build_topology("conv_net")
    graph = to_nir(spec, module)
    node = graph.nodes["conv1"]
    node.weight = node.weight + 0.5
    return graph


def test_roundtrip_mismatched_graph_exits_nonzero() -> None:
    """A graph that no longer matches the module exits non-zero."""
    report = target_cli.roundtrip_report("conv_net", graph=_perturbed_graph())
    assert report["identical"] is False
    assert target_cli.roundtrip_exit(report) == 1


def test_main_roundtrip_prints_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``roundtrip`` prints the fidelity report and exits zero."""
    assert verify.main(["roundtrip", "--topology", "conv_net"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["identical"] is True


def test_ingest_summary_runs_external_graph(tmp_path: Path) -> None:
    """A saved graph ingests through the interpreter with named traces."""
    spec, module = build_topology("conv_net")
    path = tmp_path / "graph.json"
    save_graph(to_nir(spec, module), str(path))
    summary = target_cli.ingest_summary(str(path), "conv_net")
    assert summary["steps"] == target_cli.STEPS
    assert summary["spike_nodes"]
    assert json.dumps(summary)


def test_ingest_missing_file_reports_typed_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing graph file becomes a message, not a traceback."""
    args = ["ingest", "--file", "missing.json", "--topology", "conv_net"]
    assert verify.main(args) == 1
    assert "not found" in capsys.readouterr().out


def test_ingest_malformed_file_reports_typed_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unreadable graph file becomes a message, not a traceback."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    args = ["ingest", "--file", str(bad), "--topology", "conv_net"]
    assert verify.main(args) == 1
    assert "malformed" in capsys.readouterr().out
