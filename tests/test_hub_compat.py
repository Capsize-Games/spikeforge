"""Compatibility verdicts: exact, mappable, and named mismatches."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

from spikeforge.nir_bridge import save_graph, to_nir
from spikeforge.topology.registry import build_topology
from spikeforge_hub import compat
from spikeforge_hub.inspect import inspect_artifact

pytest.importorskip("nir")


def _graph_file(
    tmp_path: Path, topology: str = "fc_small", perturb: bool = False
) -> str:
    """Save a preset's graph, optionally perturbing a structural shape."""
    spec, module = build_topology(topology)
    graph = to_nir(spec, module)
    if perturb:
        graph.nodes["fc1"].weight = np.zeros((5, 784), np.float32)
    path = tmp_path / f"{topology}.json"
    save_graph(graph, str(path))
    return str(path)


def _verdict(
    path: str, topology: Optional[str] = None
) -> compat.CompatibilityVerdict:
    """Classify the artifact at ``path``."""
    return compat.classify(inspect_artifact(path), topology)


def test_exact_graph_matches_its_preset(tmp_path: Path) -> None:
    """A preset's own graph classifies as an exact match."""
    verdict = _verdict(_graph_file(tmp_path))
    assert verdict.verdict == compat.EXACT
    assert verdict.topology == "fc_small"
    assert verdict.compatible is True
    assert verdict.mismatches == ()


def test_perturbed_graph_is_named_incompatible(tmp_path: Path) -> None:
    """A structural perturbation is rejected with the stage named."""
    verdict = _verdict(_graph_file(tmp_path, perturb=True))
    assert verdict.verdict == compat.INCOMPATIBLE
    assert verdict.compatible is False
    stages = {item["stage"] for item in verdict.mismatches}
    assert "fc1" in stages
    assert any(item["reason"] == "params" for item in verdict.mismatches)


def test_restricting_to_the_wrong_topology_rejects(tmp_path: Path) -> None:
    """Forcing an unrelated preset rejects an otherwise mappable graph."""
    verdict = _verdict(_graph_file(tmp_path), topology="conv_net")
    assert verdict.verdict == compat.INCOMPATIBLE


def test_state_dict_maps_onto_its_preset(tmp_path: Path) -> None:
    """A matching state dict is mappable with a key-to-stage mapping."""
    module = build_topology("fc_small")[1]
    path = tmp_path / "weights.pt"
    torch.save(module.state_dict(), str(path))
    verdict = _verdict(str(path))
    assert verdict.verdict == compat.MAPPABLE
    assert verdict.topology == "fc_small"
    assert verdict.mapping["fc1.weight"] == "fc1"


def test_state_dict_missing_key_is_incompatible(tmp_path: Path) -> None:
    """A state dict missing a weight is rejected with that key named."""
    module = build_topology("fc_small")[1]
    state = dict(module.state_dict())
    state.pop("fc1.weight")
    path = tmp_path / "weights.pt"
    torch.save(state, str(path))
    verdict = _verdict(str(path))
    assert verdict.verdict == compat.INCOMPATIBLE
    assert any(
        item["stage"] == "fc1" and item["reason"] == "missing-key"
        for item in verdict.mismatches
    )


def test_framework_weights_has_no_mapping(tmp_path: Path) -> None:
    """Opaque framework weights are rejected with an explanatory note."""
    path = tmp_path / "blob.bin"
    path.write_bytes(b"xx")
    verdict = _verdict(str(path))
    assert verdict.verdict == compat.INCOMPATIBLE
    assert verdict.notes


def test_verdict_is_json_serialisable(tmp_path: Path) -> None:
    """The verdict survives ``json.dumps`` for the WebSocket payload."""
    assert isinstance(
        json.dumps(_verdict(_graph_file(tmp_path)).to_dict()), str
    )
