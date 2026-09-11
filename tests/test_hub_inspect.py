"""Artifact inspection: kind detection and honest rejection."""

from pathlib import Path
from typing import Any

import pytest
import torch

from spikeforge_hub import cache
from spikeforge_hub.catalog import get
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubArtifactError
from spikeforge_hub.inspect import (
    FRAMEWORK_WEIGHTS,
    NIR_GRAPH,
    STATE_DICT,
    inspect_artifact,
    normalize_nodes,
    resolve_path,
)

pytest.importorskip("nir")


def _module(topology: str = "fc_small") -> Any:
    """Build a preset and return its runnable module."""
    from spikeforge.topology.registry import build_topology

    _spec, module = build_topology(topology)
    return module


def _graph_file(tmp_path: Path, topology: str = "fc_small") -> str:
    """Save a topology's NIR graph to disk and return its path."""
    from spikeforge.nir_bridge import save_graph, to_nir
    from spikeforge.topology.registry import build_topology

    spec, module = build_topology(topology)
    path = tmp_path / "graph.json"
    save_graph(to_nir(spec, module), str(path))
    return str(path)


def test_inspect_nir_graph_names_nodes_and_edges(tmp_path: Path) -> None:
    """A persisted NIR graph reports its nodes, edges, and kind."""
    report = inspect_artifact(_graph_file(tmp_path))
    assert report.kind == NIR_GRAPH
    assert {node["name"] for node in report.nodes} >= {"fc1", "lif1", "fc2"}
    assert report.edges
    assert report.to_dict()["kind"] == NIR_GRAPH


def test_inspect_state_dict_lists_keys_and_shapes(tmp_path: Path) -> None:
    """A bare state dict reports each key with its tensor shape."""
    path = tmp_path / "weights.pt"
    torch.save(_module().state_dict(), str(path))
    report = inspect_artifact(str(path))
    assert report.kind == STATE_DICT
    keys = {item["key"]: item["shape"] for item in report.keys}
    assert keys["fc1.weight"] == [32, 784]
    assert report.nodes == ()


def test_inspect_checkpoint_unwraps_state_dict(tmp_path: Path) -> None:
    """A full checkpoint dict is unwrapped to its inner state dict."""
    path = tmp_path / "ckpt.pt"
    module = _module()
    torch.save({"state_dict": module.state_dict(), "meta": {}}, str(path))
    report = inspect_artifact(str(path))
    assert report.kind == STATE_DICT
    assert any(item["key"] == "fc2.weight" for item in report.keys)


def test_inspect_directory_is_framework_weights(tmp_path: Path) -> None:
    """A directory snapshot is opaque framework weights by name and size."""
    snapshot = tmp_path / "repo"
    snapshot.mkdir()
    (snapshot / "model.bin").write_bytes(b"1234")
    report = inspect_artifact(str(snapshot))
    assert report.kind == FRAMEWORK_WEIGHTS
    assert report.keys[0]["key"] == "model.bin"


def test_inspect_opaque_weights_file(tmp_path: Path) -> None:
    """A declared but opaque weight file is reported, not rejected."""
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"xx")
    report = inspect_artifact(str(path))
    assert report.kind == FRAMEWORK_WEIGHTS
    assert "no structural mapping" in " ".join(report.notes)


def test_inspect_missing_path_raises(tmp_path: Path) -> None:
    """A missing artifact is rejected with a typed error."""
    with pytest.raises(HubArtifactError):
        inspect_artifact(str(tmp_path / "absent.pt"))


def test_inspect_malformed_json_raises(tmp_path: Path) -> None:
    """An unreadable NIR envelope is rejected, never silently accepted."""
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(HubArtifactError):
        inspect_artifact(str(path))


def test_inspect_unrecognized_extension_raises(tmp_path: Path) -> None:
    """An unknown extension is rejected by name."""
    path = tmp_path / "model.xyz"
    path.write_bytes(b"x")
    with pytest.raises(HubArtifactError) as ctx:
        inspect_artifact(str(path))
    assert ".xyz" in str(ctx.value)


def test_resolve_path_materializes_bundled_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bundled entry is rendered to its NIR graph in the cache."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path))
    entry = get("nir/fc_small")
    assert entry is not None
    path = resolve_path(entry)
    assert Path(path).exists()
    assert inspect_artifact(path).kind == NIR_GRAPH


def test_resolve_path_requires_a_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-bundled entry with nothing cached cannot be resolved."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path))
    entry = HubEntry.from_dict(
        {
            "id": "hf/example",
            "name": "Example HF entry",
            "framework": "hf",
            "kind": "framework_weights",
            "source": "hf_repo",
            "hf_repo": "org/repo",
            "license": "Apache-2.0",
            "notes": "unit-test entry",
        }
    )
    with pytest.raises(HubArtifactError):
        resolve_path(entry)


def test_normalize_nodes_reduces_arrays_to_shapes() -> None:
    """Node parameters expose array shapes rather than raw tensors."""
    nodes = [{"name": "fc", "kind": "Affine", "params": {"weight": [[1, 2]]}}]
    assert normalize_nodes(nodes)[0]["params"]["weight"] == [1, 2]
