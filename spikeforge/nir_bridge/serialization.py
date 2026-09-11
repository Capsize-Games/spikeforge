"""Persist and reload NIR graphs with an exact, version-stamped format.

What the installed ``nir`` provides
-----------------------------------
``nir`` 1.0.8 ships a file writer/reader (``nir.write``/``nir.read``, HDF5 via
``h5py``) and the in-memory ``NIRNode.to_dict`` / ``nir.dict2NIRNode`` pair.
Its HDF5 writer cannot persist the structural graphs this project emits:
``write_recursive`` maps a ``None`` field (``Conv2d.input_shape``, an
``Input``'s shape) to ``h5py.create_dataset(data=None)``, which raises
``TypeError``. Its reader also defaults to ``type_check=True``, which would
re-infer and mutate the untyped structural graphs.

So the file container is this project's own JSON envelope, while node
semantics stay delegated to ``nir``'s own ``to_dict`` / ``dict2NIRNode``
through :mod:`spikeforge.nir_bridge.api`. The envelope is version-stamped
so the format can evolve, and numpy values keep their dtype and shape, so the
round-trip is exact.

Reader failures are translated into typed errors: a missing file raises
:class:`GraphNotFoundError`, unreadable or inconsistent content raises
:class:`MalformedGraphError`, and a node kind ``nir`` cannot rebuild raises
:class:`UnknownNodeKindError` naming the node.
"""

import json
from typing import Any, Dict, Tuple

from spikeforge.nir_bridge import api, array_codec
from spikeforge.nir_bridge.errors import (
    GraphNotFoundError,
    MalformedGraphError,
    UnknownNodeKindError,
)

#: Envelope marker identifying this project's graph files.
FORMAT_NAME = "spikeforge-nir-graph"
#: Envelope version; bump when the container changes incompatibly.
FORMAT_VERSION = 1


def _node_payload(graph: Any) -> Dict[str, Any]:
    """Return the JSON-able ``nodes``/``edges`` payload for ``graph``."""
    return {
        "nodes": {
            name: array_codec.encode(node.to_dict())
            for name, node in graph.nodes.items()
        },
        "edges": array_codec.encode([list(edge) for edge in graph.edges]),
    }


def save_graph(graph: Any, path: str) -> None:
    """Write ``graph`` to ``path`` in this project's version-stamped format."""
    envelope = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "graph": _node_payload(graph),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(envelope, handle)


def _read_envelope(path: str) -> Dict[str, Any]:
    """Return the graph payload in the envelope at ``path``."""
    try:
        with open(path, encoding="utf-8") as handle:
            envelope = json.load(handle)
    except FileNotFoundError:
        raise GraphNotFoundError(path) from None
    except (OSError, ValueError) as error:
        raise MalformedGraphError(path, str(error)) from None
    if not isinstance(envelope, dict) or envelope.get("format") != FORMAT_NAME:
        raise MalformedGraphError(path, "not an spikeforge NIR graph")
    if envelope.get("version") != FORMAT_VERSION:
        detail = f"unsupported format version {envelope.get('version')!r}"
        raise MalformedGraphError(path, detail)
    payload = envelope.get("graph")
    if not isinstance(payload, dict):
        raise MalformedGraphError(path, "envelope has no graph object")
    return payload


def _decode_nodes(path: str, nodes: Dict[str, Any]) -> Dict[str, Any]:
    """Decode node dicts, rejecting kinds ``nir`` cannot rebuild."""
    decoded = array_codec.decode(nodes)
    for name, node in decoded.items():
        if not isinstance(node, dict):
            raise MalformedGraphError(path, f"node {name!r} is not an object")
        kind = node.get("type")
        if not api.is_serializable_node(kind):
            raise UnknownNodeKindError(path, str(kind), str(name))
    return decoded


def _decode_payload(
    path: str, payload: Dict[str, Any]
) -> Tuple[Dict[str, Any], Any]:
    """Return the decoded ``(nodes, edges)`` of a graph payload."""
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, dict) or not isinstance(edges, list):
        raise MalformedGraphError(path, "graph has no nodes or edges")
    return _decode_nodes(path, nodes), array_codec.decode(edges)


def _assemble(path: str, nodes: Dict[str, Any], edges: Any) -> Any:
    """Rebuild a ``nir.NIRGraph`` from decoded nodes and edges."""
    if not api.available():
        raise MalformedGraphError(path, "the nir package is not installed")
    request = {
        "type": "NIRGraph",
        "nodes": nodes,
        "edges": edges,
        "type_check": False,
    }
    try:
        graph = api.node_from_dict(request)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        raise MalformedGraphError(path, str(error)) from None
    if graph is None:
        raise MalformedGraphError(path, "the nir package is not installed")
    return graph


def load_graph(path: str) -> Any:
    """Read the NIR graph stored at ``path``, raising typed errors on failure.

    The reloaded graph is structurally identical to the saved one: node kinds,
    parameters (including tensors) and edges are preserved exactly. Type
    checking stays disabled so an untyped structural graph is not mutated.
    """
    payload = _read_envelope(path)
    nodes, edges = _decode_payload(path, payload)
    return _assemble(path, nodes, edges)
