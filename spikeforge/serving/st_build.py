"""Build version 2 bundles from checkpoints and v1 archives."""

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from safetensors.torch import save as _st_save

from spikeforge.network import model_store
from spikeforge.neurons.registry import NEURONS
from spikeforge.serving.encode_spec import ENCODE_SPEC_VERSION, EncodeSpec
from spikeforge.serving.errors import BundleFormatError
from spikeforge.serving.safetensor_bundle import STBundle
from spikeforge.serving.st_format import checksum, dump_json, runtime_versions
from spikeforge.serving.st_verify import tensor_inventory
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec


def model_doc(
    spec_dict: Mapping[str, Any], meta: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the neuron/reset/time/readout semantics doc."""
    spec = TopologySpec.from_dict(spec_dict)
    spec.validate()
    neurons = {s.name: s.kind for s in spec.stages if s.kind in NEURONS}
    params = {s.name: dict(s.params) for s in spec.stages if s.kind in NEURONS}
    return {
        "spec": dict(spec_dict),
        "stage_neurons": neurons,
        "neuron_params": params,
        "input": spec.input,
        "output": spec.output,
        "num_steps": meta.get("num_steps"),
        "num_classes": meta.get("num_classes"),
    }


def frozen_encode(
    explicit: Optional[Mapping[str, Any]], meta: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the validated canonical encode config, never invented."""
    base = explicit if explicit is not None else meta.get("encode")
    resolved = EncodeSpec.from_mapping(base)
    resolved.validate()
    return resolved.to_dict()


def provenance_for(
    meta: Mapping[str, Any], stored: Mapping[str, Any], source: str
) -> Tuple[Dict[str, Any], List[str]]:
    """Return the provenance record and the missing-field list."""
    record = {
        "dataset": meta.get("dataset"),
        "topology": meta.get("topology"),
        "source": source,
        "checkpoint_manifest": stored.get("manifest") or {},
    }
    missing = [k for k in ("dataset", "topology") if not meta.get(k)]
    return record, missing


def expected_doc(
    expected: Optional[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[str]]:
    """Return the expected-output doc and missing-field notes."""
    if expected:
        return {"vectors": dict(expected)}, []
    return {"vectors": {}}, ["expected_outputs"]


def clean_state(state: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Return dense CPU tensors or raise a format error."""
    clean: Dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise BundleFormatError("<build>", f"non-tensor {name}")
        tensor = value.detach().cpu().contiguous()
        if tensor.is_sparse:
            raise BundleFormatError("<build>", f"sparse {name}")
        clean[str(name)] = tensor
    return clean


def manifest_head(
    meta: Mapping[str, Any], weights: Mapping[str, torch.Tensor],
    source: str, src_ver: Any,
) -> Dict[str, Any]:
    """Return the identity and inventory header of a manifest."""
    from spikeforge.serving.st_format import ST_FORMAT, ST_VERSION

    raw = _st_save({k: v.cpu().contiguous() for k, v in weights.items()})
    head = manifest_identity(meta, source, src_ver)
    head["format"] = ST_FORMAT
    head["version"] = ST_VERSION
    head["created_at"] = time.time()
    head["tensors"] = tensor_inventory(weights)
    head["model_hash"] = checksum(raw)
    return head


def manifest_identity(
    meta: Mapping[str, Any], source: str, src_ver: Any
) -> Dict[str, Any]:
    """Return the topology and version fields of a manifest."""
    return {
        "source_format": source,
        "source_version": src_ver,
        "topology": meta.get("topology"),
        "topology_params": dict(meta.get("topology_params") or {}),
        "input_mode": meta.get("input_mode", "raw"),
        "coding": meta.get("coding", meta.get("input_mode", "raw")),
        "num_steps": meta.get("num_steps"),
        "num_classes": meta.get("num_classes"),
        "device": meta.get("device"),
        "encode_spec_version": ENCODE_SPEC_VERSION,
        "library_versions": runtime_versions(),
    }


def manifest_tail(
    doc: Mapping[str, Any], encode: Mapping[str, Any],
    record: Mapping[str, Any], missing: List[str],
    expected: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return the semantic and digest tail of a new manifest."""
    return {
        "encode_config_digest": checksum(dump_json(encode)),
        "model_digest": checksum(dump_json(doc)),
        "exp_digest": checksum(dump_json(expected)),
        "provenance": dict(record),
        "provenance_digest": checksum(dump_json(record)),
        "expected_outputs": dict(expected.get("vectors") or {}),
        "missing_metadata": list(missing),
    }


def assemble(
    weights: Mapping[str, torch.Tensor], encode: Mapping[str, Any],
    preprocess: Mapping[str, Any], doc: Mapping[str, Any],
    expected: Mapping[str, Any], record: Mapping[str, Any],
    missing: List[str], meta: Mapping[str, Any], source: str, src_ver: Any,
    graph: Optional[Mapping[str, Any]],
) -> STBundle:
    """Assemble a bundle with digests from verified parts."""
    manifest = manifest_head(meta, weights, source, src_ver)
    manifest.update(manifest_tail(doc, encode, record, missing, expected))
    return STBundle(
        manifest=manifest, weights=dict(weights), encode_config=encode,
        preprocessing=dict(preprocess), model=dict(doc),
        expected=dict(expected), graph=graph,
    )


def spec_for(spec: Mapping[str, Any]) -> TopologySpec:
    """Return the validated spec for ``spec`` dict input."""
    resolved = TopologySpec.from_dict(spec)
    resolved.validate()
    return resolved


def module_for(
    spec: Mapping[str, Any], weights: Mapping[str, torch.Tensor]
) -> Tuple[TopologySpec, Any]:
    """Return the spec and module with ``weights`` loaded strictly."""
    resolved = spec_for(spec)
    module = build_module(resolved)
    module.load_state_dict(dict(weights), strict=True)
    return resolved, module


def nir_envelope(
    spec: Mapping[str, Any], weights: Mapping[str, torch.Tensor],
    want: bool,
) -> Optional[Mapping[str, Any]]:
    """Return the preserved NIR envelope or ``None`` when skipped."""
    if not want:
        return None
    from spikeforge.nir_bridge import api, serialization
    from spikeforge.nir_bridge.exporter import to_nir
    from spikeforge.serving.st_format import GRAPH_NAME

    if not api.available():
        raise BundleFormatError("<build>", "nir extra required")
    _, module = module_for(dict(spec), weights)
    with tempfile.TemporaryDirectory() as folder:
        target = os.path.join(folder, GRAPH_NAME)
        serialization.save_graph(to_nir(spec_for(spec), module), target)
        with open(target, encoding="utf-8") as handle:
            return json.load(handle)


def build_st_bundle(
    checkpoint: str, out: Optional[str] = None,
    encode_config: Optional[Mapping[str, Any]] = None,
    preprocessing: Optional[Mapping[str, Any]] = None,
    expected: Optional[Mapping[str, Any]] = None,
    include_nir: bool = False,
) -> STBundle:
    """Build a version 2 bundle from a ``.pt`` checkpoint name."""
    stored = model_store.load(checkpoint)
    meta = dict(stored.get("meta") or {})
    spec, weights = _checkpoint_parts(checkpoint, stored, meta)
    bundle = _finish(spec, weights, meta, stored, encode_config,
                     preprocessing, expected, include_nir, checkpoint)
    if out:
        bundle.save(out)
    return bundle


def _finish(
    spec: Mapping[str, Any], weights: Dict[str, torch.Tensor],
    meta: Mapping[str, Any], stored: Mapping[str, Any],
    encode_config: Optional[Mapping[str, Any]],
    preprocessing: Optional[Mapping[str, Any]],
    expected: Optional[Mapping[str, Any]], include_nir: bool,
    checkpoint: str,
) -> STBundle:
    """Assemble the finished bundle for a checkpoint build."""
    encode = frozen_encode(encode_config, meta)
    doc = model_doc(spec, meta)
    exp_doc, exp_missing = expected_doc(expected)
    record, prov_missing = provenance_for(meta, stored, f"pt:{checkpoint}")
    graph = nir_envelope(spec, weights, include_nir)
    return assemble(weights, encode, dict(preprocessing or {}), doc,
                    exp_doc, record, exp_missing + prov_missing, meta,
                    "pt-checkpoint", 1, graph)


def _checkpoint_parts(
    checkpoint: str, stored: Mapping[str, Any], meta: Mapping[str, Any]
) -> Tuple[Mapping[str, Any], Dict[str, torch.Tensor]]:
    """Return the ``(spec, weights)`` of a stored checkpoint."""
    spec = meta.get("spec")
    if not isinstance(spec, dict):
        raise BundleFormatError(checkpoint, "no topology spec")
    if stored.get("state_dict") is None:
        raise BundleFormatError(checkpoint, "no state_dict")
    return spec, clean_state(stored["state_dict"])
