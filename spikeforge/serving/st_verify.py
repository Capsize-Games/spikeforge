"""Verification for SafeTensors bundles: sums, hashes, tensors, meta."""

from typing import Any, Dict, Mapping

import torch

from spikeforge.neurons.registry import NEURONS
from spikeforge.serving.encode_spec import ENCODE_SPEC_VERSION, EncodeSpec
from spikeforge.serving.errors import (
    BundleCompatibilityError,
    BundleFormatError,
    BundleIntegrityError,
)
from spikeforge.serving.st_format import (
    EXPECTED_NAME,
    MODEL_NAME,
    ST_FORMAT,
    ST_VERSION,
    SUMS_NAME,
    WEIGHTS_NAME,
    checksum,
    parse_sums,
)
from spikeforge.topology.spec import TopologySpec


def tensor_digest(tensor: torch.Tensor) -> str:
    """Return the SHA-256 of a tensor's dtype, shape, and bytes."""
    import hashlib

    cpu = tensor.detach().cpu().contiguous()
    tag = f"{cpu.dtype}:{list(cpu.shape)}:".encode()
    try:
        raw = cpu.numpy().tobytes()
    except TypeError:
        raw = str(cpu.flatten().tolist()).encode()
    return hashlib.sha256(tag + raw).hexdigest()


def tensor_inventory(
    weights: Mapping[str, torch.Tensor],
) -> Dict[str, Dict[str, Any]]:
    """Return the canonical per-tensor dtype/shape/digest record."""
    return {
        name: {
            "dtype": str(val.dtype),
            "shape": list(val.shape),
            "sha256": tensor_digest(val),
        }
        for name, val in weights.items()
    }


def verify_sums(path: str, payloads: Mapping[str, bytes]) -> None:
    """Raise when any entry is missing, tampered, or unlisted."""
    try:
        expected = parse_sums(payloads[SUMS_NAME].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        raise BundleFormatError(path, f"sums unreadable: {err}") from None
    for name, digest in expected.items():
        if name not in payloads:
            raise BundleIntegrityError(path, name, "listed absent")
        if checksum(payloads[name]) != digest:
            raise BundleIntegrityError(path, name)
    extra = set(payloads) - set(expected) - {SUMS_NAME}
    if extra:
        raise BundleIntegrityError(path, sorted(extra)[0], "unlisted")


def check_manifest(path: str, manifest: Any) -> Mapping[str, Any]:
    """Return ``manifest`` after marker and version checks."""
    if not isinstance(manifest, dict):
        raise BundleFormatError(path, "manifest is not an object")
    if manifest.get("format") != ST_FORMAT:
        raise BundleFormatError(path, "not a safetensor bundle")
    if manifest.get("version") != ST_VERSION:
        raise BundleFormatError(path, "mixed/unsupported version")
    if not isinstance(manifest.get("tensors"), dict):
        raise BundleFormatError(path, "manifest has no inventory")
    return manifest


def verify_tensors(
    path: str, manifest: Mapping[str, Any],
    weights: Mapping[str, torch.Tensor],
) -> None:
    """Raise on name, dtype, shape, digest, or finite-value drift."""
    recorded = manifest["tensors"]
    if set(weights) != set(recorded):
        raise BundleFormatError(path, "tensor names mismatch")
    for name, tensor in weights.items():
        _verify_one(path, name, tensor, recorded[name])


def _verify_one(
    path: str, name: str, tensor: torch.Tensor, want: Mapping[str, Any],
) -> None:
    """Raise when one tensor drifts from its recorded inventory."""
    if str(tensor.dtype) != want["dtype"]:
        raise BundleFormatError(path, f"dtype drift: {name}")
    if list(tensor.shape) != want["shape"]:
        raise BundleFormatError(path, f"bad shape: {name}")
    if tensor_digest(tensor) != want["sha256"]:
        raise BundleIntegrityError(path, name)
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise BundleFormatError(path, f"non-finite: {name}")


def verify_hashes(
    path: str, manifest: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> None:
    """Raise when a recorded content digest disagrees."""
    if checksum(payloads[WEIGHTS_NAME]) != manifest.get("model_hash"):
        raise BundleIntegrityError(path, WEIGHTS_NAME)
    if checksum(payloads[MODEL_NAME]) != manifest.get("model_digest"):
        raise BundleIntegrityError(path, MODEL_NAME)
    if checksum(payloads[EXPECTED_NAME]) != manifest.get("exp_digest"):
        raise BundleIntegrityError(path, EXPECTED_NAME)


def verify_meta(
    path: str, manifest: Mapping[str, Any], model: Mapping[str, Any],
    encode: Mapping[str, Any], strict: bool,
) -> TopologySpec:
    """Raise on inconsistent semantics or spec versions."""
    try:
        spec = TopologySpec.from_dict(model["spec"])
        spec.validate()
    except (KeyError, TypeError, ValueError) as err:
        raise BundleFormatError(path, f"bad topology: {err}") from None
    _verify_semantics(path, model, spec)
    _verify_encode(path, encode, strict)
    return spec


def _verify_semantics(
    path: str, model: Mapping[str, Any], spec: TopologySpec,
) -> None:
    """Raise when neuron/input/readout metadata disagrees."""
    neurons = {s.name for s in spec.stages if s.kind in NEURONS}
    if set(model.get("stage_neurons") or {}) != neurons:
        raise BundleFormatError(path, "neuron metadata mismatch")
    if model.get("input") != spec.input:
        raise BundleFormatError(path, "input metadata mismatch")
    if model.get("output") != spec.output:
        raise BundleFormatError(path, "readout metadata mismatch")


def _verify_encode(path: str, encode: Mapping[str, Any], strict: bool) -> None:
    """Raise when the frozen encode contract is unusable."""
    parsed = EncodeSpec.from_mapping(encode or {})
    try:
        parsed.validate()
    except Exception as err:
        raise BundleFormatError(path, f"bad encode: {err}") from None
    if strict and parsed.spec_version != ENCODE_SPEC_VERSION:
        raise BundleCompatibilityError(path, "encode version drift")
