"""Build a :class:`DeploymentBundle` from a saved checkpoint."""

from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from spikeforge.network import model_store
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle_graph import _nir_envelope
from spikeforge.serving.bundle_read import _versions
from spikeforge.serving.bundle_types import DeploymentBundle
from spikeforge.serving.bundle_weights import _prepare_weights
from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.serving.errors import BundleFormatError

#: (meta, spec, provenance, encode, preprocess, stored, encoding,
#: pruning, resolved) staged by :func:`_bundle_inputs` for assembly.
_BundleInputs = Tuple[
    Dict[str, Any],
    Any,
    Any,
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Dict[str, torch.Tensor],
]


def _meta_input_size(meta: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    """Return the frozen sensor ``(H, W)`` geometry a checkpoint records.

    The normalised ``encode_spec`` a trained checkpoint stores is preferred;
    a legacy checkpoint may instead carry an ``input_size`` on the meta or the
    topology params. Only an explicit pair is a geometry, so a flat feature
    count (which is not an ``(H, W)`` sensor) is ignored.
    """
    frozen = meta.get("encode_spec") or {}
    candidates = (
        frozen.get("input_size"),
        meta.get("input_size"),
        (meta.get("topology_params") or {}).get("input_size"),
    )
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
    return None


def _checkpoint_parts(
    checkpoint: str,
) -> Any:
    """Return ``(meta, spec, state_dict, provenance)`` from a checkpoint."""
    stored = model_store.load(checkpoint)
    meta = dict(stored.get("meta") or {})
    spec = meta.get("spec")
    if not isinstance(spec, dict):
        raise BundleFormatError(checkpoint, "checkpoint has no topology spec")
    state_dict = stored.get("state_dict")
    if state_dict is None:
        raise BundleFormatError(checkpoint, "checkpoint has no state_dict")
    return meta, spec, state_dict, dict(stored.get("manifest") or {})


def _frozen_encode(
    encode_config: Optional[Mapping[str, Any]], meta: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the canonical encode config frozen into a bundle.

    An explicit ``encode_config`` wins; otherwise the checkpoint's own
    ``meta['encode']`` is used. Either way the config is normalised through
    :class:`EncodeSpec` so the stored JSON always carries ``spec_version``,
    and the sensor geometry is injected when the checkpoint knows it but the
    config does not.
    """
    base = encode_config if encode_config is not None else meta.get("encode")
    resolved = EncodeSpec.from_mapping(base)
    if resolved.input_size is None:
        size = _meta_input_size(meta)
        if size is not None:
            resolved = replace(resolved, input_size=size)
    resolved.validate()
    return resolved.to_dict()


def _manifest_source_fields(
    spec: Any, meta: Dict[str, Any], provenance: Any, encode: Dict[str, Any]
) -> Dict[str, Any]:
    """Return the checkpoint/encode-derived manifest fields."""
    return {
        "spec": spec,
        "meta": meta,
        "provenance": provenance,
        "versions": _versions(),
        "encode_config": encode,
        "encode_spec_version": encode["spec_version"],
    }


def _manifest_option_fields(
    preprocess: Dict[str, Any],
    label_map: Optional[Mapping[int, str]],
    expected_metrics: Optional[Mapping[str, Any]],
    protocol_version: Optional[str],
    encoding: Optional[Dict[str, Any]],
    pruning: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the caller-supplied option fields of the manifest."""
    return {
        "preprocessing": preprocess,
        "label_map": dict(label_map or {}),
        "expected_metrics": dict(expected_metrics or {}),
        "protocol_version": protocol_version,
        "weights_encoding": encoding,
        "pruning": pruning,
    }


def _build_manifest(
    spec: Any, meta: Dict[str, Any], provenance: Any, encode: Dict[str, Any],
    preprocess: Dict[str, Any], label_map: Optional[Mapping[int, str]],
    expected_metrics: Optional[Mapping[str, Any]],
    protocol_version: Optional[str], encoding: Optional[Dict[str, Any]],
    pruning: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the manifest block for a new bundle."""
    fields = _manifest_source_fields(spec, meta, provenance, encode)
    fields.update(
        _manifest_option_fields(
            preprocess, label_map, expected_metrics, protocol_version,
            encoding, pruning,
        )
    )
    return bm.new_manifest(**fields)


def _finalize_bundle(
    manifest: Dict[str, Any], stored: Dict[str, Any], encode: Dict[str, Any],
    preprocess: Dict[str, Any], graph: Optional[Any], out: Optional[str],
    encoding: Optional[Dict[str, Any]],
) -> DeploymentBundle:
    """Construct the bundle and, when ``out`` is given, save it."""
    bundle = DeploymentBundle(
        manifest=manifest,
        weights=stored,
        encode_config=encode,
        preprocessing=preprocess,
        graph=graph,
        path=out,
        weights_encoding=encoding,
    )
    if out:
        bundle.save(out)
    return bundle


def _bundle_inputs(
    checkpoint: str, encode_config: Optional[Mapping[str, Any]],
    preprocessing: Optional[Mapping[str, Any]], compress: Optional[str],
    compress_bits: int, prune_sparsity: Optional[float],
    prune_strategy: str,
) -> _BundleInputs:
    """Stage the checkpoint/encode/weight parts for assembly.

    ``compress``/``prune_sparsity`` name the optional weight transform.
    """
    meta, spec, state_dict, provenance = _checkpoint_parts(checkpoint)
    encode = _frozen_encode(encode_config, meta)
    preprocess = dict(preprocessing or {})
    stored, encoding, pruning, resolved = _prepare_weights(
        state_dict, compress, compress_bits, prune_sparsity, prune_strategy,
    )
    return (
        meta, spec, provenance, encode, preprocess, stored, encoding,
        pruning, resolved)


def _bundle_assembled(
    inputs: _BundleInputs, label_map: Optional[Mapping[int, str]],
    expected_metrics: Optional[Mapping[str, Any]],
    protocol_version: Optional[str], include_nir: bool,
    checkpoint: str, out: Optional[str],
) -> DeploymentBundle:
    """Assemble the staged parts into a bundle, saving it to ``out``.

    ``include_nir`` adds the graph envelope (needs the ``nir`` extra).
    """
    (meta, spec, provenance, encode, preprocess, stored, encoding,
     pruning, resolved) = inputs
    manifest = _build_manifest(
        spec, meta, provenance, encode, preprocess, label_map,
        expected_metrics, protocol_version, encoding, pruning)
    graph = None
    if include_nir:
        graph = _nir_envelope(spec, resolved, checkpoint)
    return _finalize_bundle(
        manifest, stored, encode, preprocess, graph, out, encoding)


def build(checkpoint: str, out: Optional[str] = None,
    encode_config: Optional[Mapping[str, Any]] = None,
    preprocessing: Optional[Mapping[str, Any]] = None,
    label_map: Optional[Mapping[int, str]] = None,
    expected_metrics: Optional[Mapping[str, Any]] = None,
    include_nir: bool = False, protocol_version: Optional[str] = None,
    compress: Optional[str] = None, compress_bits: int = 8,
    prune_sparsity: Optional[float] = None,
    prune_strategy: str = "unstructured",
) -> DeploymentBundle:
    """Build a :class:`DeploymentBundle` from a saved checkpoint."""
    inputs = _bundle_inputs(
        checkpoint, encode_config, preprocessing, compress,
        compress_bits, prune_sparsity, prune_strategy,
    )
    return _bundle_assembled(
        inputs, label_map, expected_metrics, protocol_version,
        include_nir, checkpoint, out,
    )
