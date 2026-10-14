"""Read, verify, and compatibility-check a ``.spkf`` bundle archive."""

import io
import json
import os
import platform
import zipfile
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.encode_spec import (
    ENCODE_SPEC_VERSION,
    EncodeSpec,
)
from spikeforge.serving.errors import (
    BundleCompatibilityError,
    BundleFormatError,
    BundleIntegrityError,
    BundleNotFoundError,
    EncodeSpecError,
)
from spikeforge.tracking.versions import library_versions


def _major_minor(version: Any) -> Optional[str]:
    """Return the ``major.minor`` prefix of ``version``, or None."""
    if not version:
        return None
    parts = str(version).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(version)


def _versions() -> Dict[str, Optional[str]]:
    """Return the runtime versions recorded in a bundle manifest."""
    versions: Dict[str, Optional[str]] = dict(library_versions())
    versions["python"] = platform.python_version()
    return versions


def _load_json(path: str, payload: bytes, name: str) -> Any:
    """Return the JSON value in ``payload``, or raise a typed error."""
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleFormatError(
            path, f"{name} is not valid JSON: {error}"
        ) from None


def _load_weights(path: str, payload: bytes) -> Mapping[str, torch.Tensor]:
    """Return the state dict in ``payload``, or raise a typed error."""
    try:
        weights = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise BundleFormatError(
            path, f"weights are not loadable: {error}"
        ) from None
    if not isinstance(weights, dict):
        raise BundleFormatError(path, "weights payload is not a state dict")
    return weights


def _load_preprocessing(path: str, payloads: Mapping[str, bytes]) -> Any:
    """Return the bundle's preprocessing block, or raise a typed error."""
    return _load_json(path, payloads[bm.PREPROCESSING_NAME], "preprocessing")


def _verify_integrity(path: str, payloads: Mapping[str, bytes]) -> None:
    """Raise when any entry is missing, tampered, or unlisted."""
    try:
        expected = bm.parse_sums(payloads[bm.SUMS_NAME].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise BundleFormatError(
            path, f"SHA256SUMS is unreadable: {error}"
        ) from None
    for name, digest in expected.items():
        if name not in payloads:
            raise BundleIntegrityError(path, name, "listed but absent")
        if bm.checksum(payloads[name]) != digest:
            raise BundleIntegrityError(path, name)
    unlisted = set(payloads) - set(expected) - {bm.SUMS_NAME}
    if unlisted:
        raise BundleIntegrityError(
            path, sorted(unlisted)[0], "present but not checksummed"
        )


def _check_compatibility(
    path: str, recorded: Mapping[str, Any], strict: bool
) -> None:
    """Raise when the runtime disagrees with a bundle under ``strict``."""
    if not strict:
        return
    current = _versions()
    for key in ("torch", "python"):
        want = _major_minor(recorded.get(key))
        have = _major_minor(current.get(key))
        if want and have and want != have:
            raise BundleCompatibilityError(
                path, f"{key} {have!r} does not match the bundle's {want!r}"
            )


def _parse_encode_spec(path: str, encode_config: Any) -> EncodeSpec:
    """Parse and validate an encode spec, wrapping a parse error typed."""
    spec = EncodeSpec.from_mapping(encode_config or {})
    try:
        spec.validate()
    except EncodeSpecError as error:
        raise BundleFormatError(
            path, f"invalid encode spec: {error}"
        ) from None
    return spec


def _check_encode_version(
    path: str, recorded: Optional[Any], strict: bool
) -> None:
    """Raise when ``recorded`` mismatches the runtime's spec version."""
    if recorded is None:
        return
    if int(recorded) != ENCODE_SPEC_VERSION and strict:
        raise BundleCompatibilityError(
            path,
            f"encode spec version {recorded!r} does not match the runtime's "
            f"{ENCODE_SPEC_VERSION!r}",
        )


def _check_encode(
    path: str, manifest: Mapping[str, Any], encode_config: Any, strict: bool
) -> EncodeSpec:
    """Validate a bundle's frozen encode spec and pin its contract version.

    An empty ``encode_config`` (a legacy in-memory bundle) resolves to the
    defaults, preserving the current behaviour. A non-empty spec must parse
    and validate, and its manifest ``encode_spec_version`` must match this
    runtime; a mismatch is refused under ``strict`` rather than re-encoded.
    """
    spec = _parse_encode_spec(path, encode_config)
    _check_encode_version(path, manifest.get("encode_spec_version"), strict)
    return spec


def _check_decompressed_size(
    path: str, archive: zipfile.ZipFile, names: Any
) -> None:
    """Raise when ``names``' total decompressed size exceeds the cap.

    Checked against each entry's recorded ``file_size`` before any entry is
    decompressed, so a small compressed payload declaring an enormous
    decompressed size (a zip bomb) is refused rather than read into memory.
    """
    total = sum(archive.getinfo(name).file_size for name in names)
    if total > bm.MAX_DECOMPRESSED_BYTES:
        raise BundleFormatError(
            path,
            f"decompressed size {total} exceeds the "
            f"{bm.MAX_DECOMPRESSED_BYTES} byte cap",
        )


def _read_archive(path: str) -> Dict[str, bytes]:
    """Return every entry of the zip at ``path``, or raise a typed error."""
    if not os.path.exists(path):
        raise BundleNotFoundError(path)
    try:
        with zipfile.ZipFile(path) as archive:
            names = {n for n in archive.namelist() if not n.endswith("/")}
            missing = [n for n in bm.REQUIRED_ENTRIES if n not in names]
            if missing:
                raise BundleFormatError(
                    path, f"missing required entries: {missing}"
                )
            _check_decompressed_size(path, archive, names)
            return {name: archive.read(name) for name in names}
    except zipfile.BadZipFile as error:
        raise BundleFormatError(
            path, f"not a zip archive: {error}"
        ) from None


def _read_bundle_parts(
    path: str, strict: bool
) -> Tuple[Mapping[str, Any], Any, Optional[Any], Dict[str, bytes]]:
    """Read and validate every manifest-adjacent part of a bundle archive."""
    payloads = _read_archive(path)
    _verify_integrity(path, payloads)
    manifest = bm.validate_manifest(
        _load_json(path, payloads[bm.MANIFEST_NAME], "manifest"), path
    )
    _check_compatibility(path, manifest.get("library_versions") or {}, strict)
    encode_config = _load_json(
        path, payloads[bm.ENCODE_NAME], "encode_config"
    )
    _check_encode(path, manifest, encode_config, strict)
    graph = None
    if bm.GRAPH_NAME in payloads:
        graph = _load_json(path, payloads[bm.GRAPH_NAME], "graph")
    return manifest, encode_config, graph, payloads
