"""The inspect -> compat -> promote funnel and the weight loader."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from spikeforge.network import model_store
from spikeforge.nir_bridge import save_graph, to_nir
from spikeforge.topology.registry import build_topology
from spikeforge_hub import cache
from spikeforge_hub.errors import HubArtifactError
from spikeforge_hub.import_model import import_model
from spikeforge_hub.weight_map import load_weights

pytest.importorskip("nir")


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the hub cache and model store into a temporary directory."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path / "hub"))
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path / "models"))
    return tmp_path


def _graph_file(tmp_path: Path, perturb: bool = False) -> str:
    """Save an fc_small graph, optionally with a perturbed fc1 weight."""
    spec, module = build_topology("fc_small")
    graph = to_nir(spec, module)
    if perturb:
        graph.nodes["fc1"].weight = np.zeros((5, 784), np.float32)
    path = tmp_path / "graph.json"
    save_graph(graph, str(path))
    return str(path)


def test_bundled_entry_promotes_with_validation(sandbox: Path) -> None:
    """A bundled NIR preset imports exact, validates, and promotes."""
    result = import_model("nir/fc_small")
    assert result["verdict"]["verdict"] == "exact"
    assert result["promoted"] is True
    assert result["validation"]["within_tolerance"] is True
    assert os.path.exists(result["destination"])


def test_import_records_hub_provenance(sandbox: Path) -> None:
    """The promoted checkpoint's meta names the hub source and topology."""
    result = import_model("nir/fc_small")
    assert result["meta"]["source"] == "hub"
    assert result["meta"]["hub_id"] == "nir/fc_small"
    assert result["meta"]["topology"] == "fc_small"


def test_perturbed_graph_is_rejected_by_stage(sandbox: Path) -> None:
    """A structurally perturbed artifact is not promoted and names fc1."""
    result = import_model(path=_graph_file(sandbox, perturb=True))
    assert result["promoted"] is False
    assert result["verdict"]["verdict"] == "incompatible"
    stages = {item["stage"] for item in result["verdict"]["mismatches"]}
    assert "fc1" in stages
    assert result["nir_ingestable"] is True


def test_framework_weights_are_rejected(sandbox: Path) -> None:
    """Opaque weights are rejected and are not NIR-ingestable."""
    path = sandbox / "blob.bin"
    path.write_bytes(b"xx")
    result = import_model(path=str(path))
    assert result["promoted"] is False
    assert result["nir_ingestable"] is False


def test_promote_false_skips_saving(sandbox: Path) -> None:
    """The funnel can classify without promoting when asked."""
    result = import_model("nir/fc_small", promote=False)
    assert result["verdict"]["verdict"] == "exact"
    assert result["promoted"] is False
    assert result["destination"] is None


def test_unknown_entry_raises() -> None:
    """An unknown catalog id is a typed error, not a silent miss."""
    with pytest.raises(HubArtifactError):
        import_model("does/not-exist")


def test_missing_path_raises() -> None:
    """A missing path is a typed error."""
    with pytest.raises(HubArtifactError):
        import_model(path="missing-artifact.pt")


def test_load_weights_from_graph_matches_module() -> None:
    """A graph's weights map onto a freshly built preset exactly."""
    spec, module = build_topology("fc_small")
    _other_spec, target = build_topology("fc_small")
    result = load_weights(target, graph=to_nir(spec, module))
    assert result.loaded is True
    expected = torch.as_tensor(to_nir(spec, module).nodes["fc1"].weight)
    assert torch.allclose(target.fc1.weight, expected)


def test_load_weights_reports_missing_key() -> None:
    """A state dict missing a weight is refused with the key named."""
    _spec, module = build_topology("fc_small")
    state = dict(module.state_dict())
    state.pop("fc2.weight")
    result = load_weights(module, state_dict=state)
    assert result.loaded is False
    assert "fc2.weight" in result.missing


def test_load_weights_reports_shape_mismatch() -> None:
    """A wrong-shaped tensor is refused rather than broadcast on load."""
    _spec, module = build_topology("fc_small")
    state = dict(module.state_dict())
    state["fc1.weight"] = torch.zeros(5, 784)
    result = load_weights(module, state_dict=state)
    assert result.loaded is False
    assert "fc1.weight" in result.mismatched


def test_load_weights_needs_exactly_one_source() -> None:
    """Supplying no source (or both) is a programming error."""
    _spec, module = build_topology("fc_small")
    with pytest.raises(ValueError):
        load_weights(module)
