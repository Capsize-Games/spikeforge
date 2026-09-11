"""Constants and validation for the deployment-bundle format.

A ``.spkf`` bundle is a zip with a fixed entry set. This module owns the entry
names, the format marker and version, the manifest assembly, and the checksum
text format, so the builder and the reader cannot disagree about what a bundle
contains.
"""

import hashlib
import json
import time
from typing import Any, Dict, Mapping, Optional

from spikeforge.serving.encode_spec import ENCODE_SPEC_VERSION
from spikeforge.serving.errors import BundleFormatError

#: Marker identifying a deployment bundle.
BUNDLE_FORMAT = "spikeforge-deployment-bundle"
#: Container version; bump when the entry set changes incompatibly.
BUNDLE_VERSION = 1
#: Conventional file suffix for a bundle.
BUNDLE_SUFFIX = ".spkf"

MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.pt"
ENCODE_NAME = "encode_config.json"
PREPROCESSING_NAME = "preprocessing.json"
GRAPH_NAME = "graph.nir.json"
SUMS_NAME = "SHA256SUMS"

#: Entries every bundle must carry.
REQUIRED_ENTRIES = (
    MANIFEST_NAME,
    WEIGHTS_NAME,
    ENCODE_NAME,
    PREPROCESSING_NAME,
    SUMS_NAME,
)
#: Entries a bundle may additionally carry.
OPTIONAL_ENTRIES = (GRAPH_NAME,)


def checksum(payload: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def sums_text(entries: Mapping[str, bytes]) -> str:
    """Return the ``SHA256SUMS`` body for ``entries`` (GNU two-space form)."""
    return "".join(
        f"{checksum(payload)}  {name}\n" for name, payload in entries.items()
    )


def parse_sums(text: str) -> Dict[str, str]:
    """Return ``{name: sha256}`` parsed from a ``SHA256SUMS`` body."""
    parsed: Dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"malformed checksum line: {stripped!r}")
        parsed[parts[1].strip()] = parts[0].strip().lower()
    return parsed


def dump_json(payload: Any) -> bytes:
    """Return ``payload`` as stable, indented UTF-8 JSON bytes."""
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def validate_manifest(manifest: Any, path: str) -> Mapping[str, Any]:
    """Return ``manifest`` after checking the marker, version, and spec."""
    if not isinstance(manifest, dict):
        raise BundleFormatError(path, "manifest is not an object")
    marker = manifest.get("format")
    if marker != BUNDLE_FORMAT:
        raise BundleFormatError(
            path, f"not a spikeforge deployment bundle: {marker!r}"
        )
    if manifest.get("version") != BUNDLE_VERSION:
        detail = f"unsupported bundle version {manifest.get('version')!r}"
        raise BundleFormatError(path, detail)
    if not isinstance(manifest.get("spec"), dict):
        raise BundleFormatError(path, "manifest has no topology spec")
    return manifest


def new_manifest(
    spec: Mapping[str, Any],
    meta: Mapping[str, Any],
    provenance: Mapping[str, Any],
    versions: Mapping[str, Any],
    encode_config: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    label_map: Mapping[str, Any],
    expected_metrics: Mapping[str, Any],
    protocol_version: Optional[str],
    encode_spec_version: Optional[int] = None,
    created_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble the self-describing manifest written into a bundle.

    ``encode_spec_version`` records which encode contract the ``encode_config``
    was frozen under; it defaults to the runtime's current version and is
    additive to the entry set, so older readers ignore it.
    """
    return {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "created_at": time.time() if created_at is None else float(created_at),
        "protocol_version": protocol_version,
        "encode_spec_version": (
            ENCODE_SPEC_VERSION
            if encode_spec_version is None
            else int(encode_spec_version)
        ),
        "library_versions": dict(versions),
        "spec": dict(spec),
        "topology": meta.get("topology"),
        "topology_params": dict(meta.get("topology_params") or {}),
        "input_mode": meta.get("input_mode", "raw"),
        "coding": meta.get("coding", meta.get("input_mode", "raw")),
        "num_steps": meta.get("num_steps"),
        "num_classes": meta.get("num_classes"),
        "device": meta.get("device"),
        "label_map": {str(key): val for key, val in dict(label_map).items()},
        "expected_metrics": dict(expected_metrics),
        "encode_config": dict(encode_config),
        "preprocessing": dict(preprocessing),
        "provenance": dict(provenance),
    }
