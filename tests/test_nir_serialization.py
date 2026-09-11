"""Preset graph round-trips and typed graph-file errors."""

import json
from typing import Any, Dict, List, Tuple

import pytest
import torch

from spikeforge.nir_bridge import array_codec
from spikeforge.nir_bridge.errors import (
    GraphNotFoundError,
    MalformedGraphError,
    UnknownNodeKindError,
)
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.nir_bridge.ops_registry import SUPPORTED_KINDS
from spikeforge.nir_bridge.serialization import (
    FORMAT_NAME,
    FORMAT_VERSION,
    load_graph,
    save_graph,
)
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec
from spikeforge_targets.primitives import EMITTED_PRIMITIVES

pytest.importorskip("nir")

_CASES: Dict[str, Tuple[TopologySpec, Tuple[int, ...]]] = {
    "fc_legacy": (
        presets.fc_legacy(hidden=8, beta=0.5, num_classes=3, input_size=5),
        (6, 2, 5),
    ),
    "fc_small": (
        presets.fc_small(hidden=8, beta=0.9, num_classes=3, input_size=12),
        (6, 2, 1, 3, 4),
    ),
    "conv_net": (
        presets.conv_net(
            in_channels=1, channels=2, num_classes=3, input_size=8
        ),
        (6, 2, 1, 8, 8),
    ),
    "recurrent_net": (
        presets.recurrent_net(hidden=6, beta=0.9, num_classes=3, input_size=4),
        (6, 2, 4),
    ),
}
_NAMES = sorted(_CASES)


def _graph(spec: TopologySpec) -> Any:
    """Return ``spec``'s exported graph with its built module weights."""
    return to_nir(spec, build_module(spec))


def _spikes(shape: Tuple[int, ...]) -> torch.Tensor:
    """Return a fixed-seed spike train of ``shape``."""
    torch.manual_seed(0)
    return torch.rand(*shape)


def _kinds(graph: Any) -> List[str]:
    """Return the graph's node kind names in node order."""
    return [type(node).__name__ for node in graph.nodes.values()]


def _params(graph: Any) -> Dict[str, Any]:
    """Return each node's serialized parameters as JSON-able data."""
    return {
        name: array_codec.encode(node.to_dict())
        for name, node in graph.nodes.items()
    }


@pytest.mark.parametrize("name", _NAMES)
def test_reloaded_graph_matches_original(name: str, tmp_path: Any) -> None:
    """Node kinds, edges and parameters survive the round-trip exactly."""
    spec, _ = _CASES[name]
    graph = _graph(spec)
    path = str(tmp_path / f"{name}.nir.json")
    save_graph(graph, path)
    loaded = load_graph(path)
    assert _kinds(loaded) == _kinds(graph)
    assert list(loaded.edges) == list(graph.edges)
    assert _params(loaded) == _params(graph)


@pytest.mark.parametrize("name", _NAMES)
def test_reloaded_interpretation_is_bit_exact(
    name: str, tmp_path: Any
) -> None:
    """Interpreting the reloaded graph reproduces the original bit for bit."""
    spec, shape = _CASES[name]
    graph = _graph(spec)
    spikes = _spikes(shape)
    path = str(tmp_path / f"{name}.nir.json")
    save_graph(graph, path)
    original = NirInterpreter(graph).run(spikes)
    reloaded = NirInterpreter(load_graph(path)).run(spikes)
    assert reloaded.steps == original.steps
    assert torch.equal(reloaded.readout, original.readout)
    assert set(reloaded.spikes) == set(original.spikes)
    for key, train in original.spikes.items():
        assert torch.equal(reloaded.spikes[key], train)
    for key, trace in original.membranes.items():
        assert torch.equal(reloaded.membranes[key], trace)


def test_saved_file_is_a_version_stamped_json_envelope(tmp_path: Any) -> None:
    """The persisted artifact is a plain JSON envelope with a format stamp."""
    spec, _ = _CASES["fc_legacy"]
    path = tmp_path / "graph.nir.json"
    save_graph(_graph(spec), str(path))
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["format"] == FORMAT_NAME
    assert envelope["version"] == FORMAT_VERSION
    assert set(envelope["graph"]) == {"nodes", "edges"}


def test_interpreter_covers_every_emitted_primitive() -> None:
    """Every primitive a target may receive is implemented, never skipped."""
    assert EMITTED_PRIMITIVES <= SUPPORTED_KINDS


def test_missing_file_raises_graph_not_found(tmp_path: Any) -> None:
    """A path that does not exist raises the typed not-found error."""
    path = str(tmp_path / "missing.nir.json")
    with pytest.raises(GraphNotFoundError) as excinfo:
        load_graph(path)
    assert excinfo.value.path == path


def test_non_json_content_raises_malformed(tmp_path: Any) -> None:
    """Unreadable content raises the typed malformed error."""
    path = tmp_path / "broken.nir.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(MalformedGraphError):
        load_graph(str(path))


def test_foreign_envelope_raises_malformed(tmp_path: Any) -> None:
    """Valid JSON that is not our envelope raises the malformed error."""
    path = tmp_path / "other.json"
    path.write_text(
        json.dumps({"format": "something-else", "version": 1}),
        encoding="utf-8",
    )
    with pytest.raises(MalformedGraphError):
        load_graph(str(path))


def test_unknown_version_raises_malformed(tmp_path: Any) -> None:
    """A future envelope version raises the malformed error, not a crash."""
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION + 1,
        "graph": {"nodes": {}, "edges": []},
    }
    path = tmp_path / "future.nir.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MalformedGraphError):
        load_graph(str(path))


def test_unknown_node_kind_names_the_node(tmp_path: Any) -> None:
    """An unbuildable node kind raises a typed error naming the node."""
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "graph": {"nodes": {"banana": {"type": "Banana"}}, "edges": []},
    }
    path = tmp_path / "unknown.nir.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UnknownNodeKindError) as excinfo:
        load_graph(str(path))
    assert excinfo.value.kind == "Banana"
    assert excinfo.value.name == "banana"
