"""Container constants and archive IO for SafeTensors bundles."""

import hashlib
import json
import os
import platform
import zipfile
from typing import Any, Dict, Mapping, Optional

import torch
from safetensors.torch import load as _st_load

from spikeforge.serving.errors import BundleFormatError, BundleNotFoundError
from spikeforge.tracking.versions import library_versions

#: Marker identifying a SafeTensors bundle.
ST_FORMAT = "spikeforge-safetensor-bundle"
#: Container version; v1 archives import but never load here.
ST_VERSION = 2
#: Conventional file suffix for a version 2 bundle.
ST_SUFFIX = ".stbk"

MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.safetensors"
ENCODE_NAME = "encode_config.json"
PREPROCESSING_NAME = "preprocessing.json"
MODEL_NAME = "model.json"
EXPECTED_NAME = "expected.json"
GRAPH_NAME = "graph.nir.json"
SUMS_NAME = "SHA256SUMS"

#: Entries every version 2 bundle must carry.
REQUIRED_ENTRIES = (
    MANIFEST_NAME,
    WEIGHTS_NAME,
    ENCODE_NAME,
    PREPROCESSING_NAME,
    MODEL_NAME,
    EXPECTED_NAME,
    SUMS_NAME,
)
#: Entries a version 2 bundle may additionally carry.
OPTIONAL_ENTRIES = (GRAPH_NAME,)

#: Refuse archives whose entries decompress past this total.
ST_MAX_BYTES = 256 * 1024 * 1024


def checksum(payload: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def sums_text(entries: Mapping[str, bytes]) -> str:
    """Return the ``SHA256SUMS`` body for ``entries``."""
    return "".join(f"{checksum(p)}  {n}\n" for n, p in entries.items())


def parse_sums(text: str) -> Dict[str, str]:
    """Return ``{name: sha256}`` parsed from a sums body."""
    parsed: Dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"bad sums line: {stripped!r}")
        parsed[parts[1].strip()] = parts[0].strip().lower()
    return parsed


def dump_json(payload: Any) -> bytes:
    """Return ``payload`` as stable UTF-8 JSON bytes."""
    return json.dumps(payload, indent=2, sort_keys=True).encode()


def runtime_versions() -> Dict[str, Optional[str]]:
    """Return the runtime versions recorded in a manifest."""
    versions: Dict[str, Optional[str]] = dict(library_versions())
    versions["python"] = platform.python_version()
    return versions


def load_json(path: str, payload: bytes, name: str) -> Any:
    """Return the JSON value in ``payload`` or raise a typed error."""
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        raise BundleFormatError(path, f"{name} is not JSON: {err}") from None


def load_weights(path: str, payload: bytes) -> Dict[str, torch.Tensor]:
    """Return the SafeTensors state dict or raise a typed error."""
    try:
        weights = _st_load(payload)
    except Exception as err:
        raise BundleFormatError(path, f"weights unloadable: {err}") from None
    if not isinstance(weights, dict) or not weights:
        raise BundleFormatError(path, "weights hold no tensors")
    return dict(weights)


def _check_size(path: str, archive: zipfile.ZipFile) -> None:
    """Raise when the archive declares past the decompressed cap."""
    names = [n for n in archive.namelist() if not n.endswith("/")]
    total = sum(archive.getinfo(n).file_size for n in names)
    if total > ST_MAX_BYTES:
        raise BundleFormatError(path, f"size {total} over cap")


def read_archive(path: str) -> Dict[str, bytes]:
    """Return every entry of the zip at ``path`` or raise typed."""
    if not os.path.exists(path):
        raise BundleNotFoundError(path)
    try:
        with zipfile.ZipFile(path) as archive:
            raw = set(archive.namelist())
            names = {n for n in raw if not n.endswith("/")}
            missing = [n for n in REQUIRED_ENTRIES if n not in names]
            if missing:
                raise BundleFormatError(path, f"missing {missing}")
            _check_size(path, archive)
            return {n: archive.read(n) for n in names}
    except zipfile.BadZipFile as err:
        raise BundleFormatError(path, f"not a zip: {err}") from None


#: JSON entries in fixed slot order for the small slot reader.
SLOT_NAMES = (
    MANIFEST_NAME,
    MODEL_NAME,
    ENCODE_NAME,
    PREPROCESSING_NAME,
    EXPECTED_NAME,
)


def slot_json(path: str, payloads: Mapping[str, bytes], slot: int) -> Any:
    """Return one JSON entry selected by ``slot`` index."""
    name = SLOT_NAMES[slot]
    return load_json(path, payloads[name], name)


def graph_entry(path: str, payloads: Mapping[str, bytes]) -> Any:
    """Return the optional preserved NIR envelope, if present."""
    return load_json(path, payloads[GRAPH_NAME], "graph")
