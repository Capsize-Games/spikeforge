"""C packages: a bundle exported as torch-free, C-loadable JSON."""

import hashlib
import json
import os
from typing import Any, Dict, List

import pytest
import torch

from spikeforge.nir_bridge.exporter import graph_summary, to_nir
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle, build
from spikeforge.serving.c_package import (
    GRAPH_FILENAME,
    MANIFEST_FILENAME,
    PACKAGE_FORMAT,
    PACKAGE_VERSION,
    SUPPORTED_NODE_KINDS,
)
from spikeforge.training.training_engine import TrainingEngine

pytest.importorskip("nir")

#: The node kinds ``nir_graph_load`` implements. Mirrors ``KIND_TABLE`` in
#: ``nir-c-runtime``'s ``src/graph.c``; the export allow-list must match it.
_C_KINDS = frozenset(
    {
        "Input",
        "Output",
        "Conv2d",
        "Affine",
        "Flatten",
        "AvgPool2d",
        "SumPool2d",
        "LI",
        "Threshold",
        "Scale",
        "Delay",
    }
)

_CKPT = "c_package_conv_ckpt"


def _checkpoint() -> str:
    """Save a small conv_net checkpoint and return its name."""
    torch.manual_seed(0)
    engine = TrainingEngine(
        dataset="mnist",
        num_steps=4,
        device="cpu",
        topology="conv_net",
        topology_params={"channels": 2, "num_classes": 3},
    )
    engine.save(_CKPT)
    return _CKPT


def _file_bundle(tmp_path: Any) -> DeploymentBundle:
    """Build a saved bundle and load it back from disk."""
    out = str(tmp_path / "model.spkf")
    build(_checkpoint(), out=out)
    return DeploymentBundle.load(out)


def _numbers(value: Any) -> List[float]:
    """Flatten nested weight lists, refusing anything but numbers."""
    if isinstance(value, bool):
        raise AssertionError("a weight holds a boolean, not a number")
    if isinstance(value, (int, float)):
        return [float(value)]
    assert isinstance(value, list), f"a weight is not a list: {value!r:.60}"
    flat: List[float] = []
    for item in value:
        flat.extend(_numbers(item))
    return flat


def _weight_shape(value: Any) -> List[int]:
    """Return the nested-list rank shape of a weight.

    Envelopes never reach here: :func:`_numbers` already refused any
    parameter that is not plain nested numbers.
    """
    shape: List[int] = []
    while isinstance(value, list):
        shape.append(len(value))
        if not value:
            break
        value = value[0]
    return shape


def assert_c_loadable(summary: Dict[str, Any]) -> None:
    """Assert ``summary`` satisfies the C loader's documented contract.

    This mirrors ``nir_graph_load`` in ``nir-c-runtime``'s ``src/graph.c``
    (``graph_summary`` shape, the ``KIND_TABLE`` allow-list, plain nested
    float arrays with the rank each kind requires, edges naming known
    nodes), so a future export change that breaks C loading fails here.
    """
    assert isinstance(summary["nodes"], list)
    assert isinstance(summary["edges"], list)
    names = set()
    for record in summary["nodes"]:
        assert record["kind"] in _C_KINDS, record["kind"]
        names.add(record["name"])
        _numbers_in_params(record)
    for edge in summary["edges"]:
        assert edge["source"] in names
        assert edge["target"] in names


def _numbers_in_params(record: Dict[str, Any]) -> None:
    """Assert one node's weight params are plain numeric arrays."""
    params = record["params"]
    for name in ("weight", "bias"):
        if params.get(name) is None:
            continue
        _numbers(params[name])
    if record["kind"] == "Conv2d":
        assert len(_weight_shape(params["weight"])) == 4
    if record["kind"] == "Affine":
        assert len(_weight_shape(params["weight"])) == 2
        assert len(_weight_shape(params["bias"])) == 1


def _graph_of(package: str) -> Dict[str, Any]:
    """Return the parsed graph file of the package at ``package``."""
    with open(os.path.join(package, GRAPH_FILENAME), encoding="utf-8") as fh:
        return json.load(fh)


def _manifest_of(package: str) -> Dict[str, Any]:
    """Return the parsed manifest file of the package at ``package``."""
    path = os.path.join(package, MANIFEST_FILENAME)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_supported_kinds_match_the_c_loader_table() -> None:
    """The export allow-list is exactly the C ``KIND_TABLE``."""
    assert SUPPORTED_NODE_KINDS == _C_KINDS


def test_export_matches_graph_summary_with_real_weights(
    tmp_path: Any,
) -> None:
    """The package graph is the trained graph, not placeholders."""
    bundle = _file_bundle(tmp_path)
    package = bundle.export_c_package(str(tmp_path / "c_pkg"))
    assert package == str(tmp_path / "c_pkg")
    module = bundle.build_module()
    expected = graph_summary(to_nir(bundle.spec, module))
    assert _graph_of(package) == expected
    peak = 0.0
    for record in expected["nodes"]:
        weight = record["params"].get("weight")
        if isinstance(weight, list):
            peak = max(peak, max(_numbers(weight), default=0.0))
    assert peak > 0.0


def test_package_manifest_records_source_identity(tmp_path: Any) -> None:
    """The manifest names the format and checksums the source bundle."""
    bundle = _file_bundle(tmp_path)
    assert bundle.path is not None
    package = bundle.export_c_package(str(tmp_path / "c_pkg"))
    manifest = _manifest_of(package)
    assert manifest["format"] == PACKAGE_FORMAT
    assert manifest["version"] == PACKAGE_VERSION
    assert manifest["graph"] == GRAPH_FILENAME
    graph = _graph_of(package)
    assert manifest["nodes"] == len(graph["nodes"])
    assert manifest["edges"] == len(graph["edges"])
    source = manifest["source"]
    assert source["bundle_path"] == bundle.path
    with open(bundle.path, "rb") as handle:
        expected = hashlib.sha256(handle.read()).hexdigest()
    assert source["bundle_sha256"] == expected
    assert source["manifest_sha256"] == bm.checksum(
        bm.dump_json(bundle.manifest)
    )


def test_in_memory_bundle_exports_without_a_file_checksum(
    tmp_path: Any,
) -> None:
    """A bundle with no file still exports with its manifest checksum."""
    bundle = build(_checkpoint())
    assert bundle.path is None
    package = bundle.export_c_package(str(tmp_path / "c_pkg"))
    source = _manifest_of(package)["source"]
    assert source["bundle_path"] is None
    assert source["bundle_sha256"] is None
    assert source["manifest_sha256"] == bm.checksum(
        bm.dump_json(bundle.manifest)
    )
    assert_c_loadable(_graph_of(package))


def test_trained_package_is_c_conformant(tmp_path: Any) -> None:
    """A trained conv_net package satisfies the C loader's contract."""
    bundle = _file_bundle(tmp_path)
    package = bundle.export_c_package(str(tmp_path / "c_pkg"))
    assert_c_loadable(_graph_of(package))


def test_compressed_bundle_exports_dequantized_floats(
    tmp_path: Any,
) -> None:
    """A compressed bundle exports plain floats, never code tensors."""
    out = str(tmp_path / "model.spkf")
    build(_checkpoint(), out=out, compress="int8")
    bundle = DeploymentBundle.load(out)
    assert bundle.weights_encoding is not None
    graph = _graph_of(bundle.export_c_package(str(tmp_path / "c_pkg")))
    module = bundle.build_module()
    assert graph == graph_summary(to_nir(bundle.spec, module))
    assert_c_loadable(graph)



