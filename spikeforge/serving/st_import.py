"""Explicit backward-compatible import of v1 bundles and .pt files.

A v1 ``.spkf`` archive (``weights.pt`` plus JSON entries) converts
into a version 2 SafeTensors bundle. The original file is never
modified; its SHA-256 and entry list are preserved in the new
provenance record. Fields version 2 requires but version 1 lacks
are classified in the import report and the manifest's
``missing_metadata`` list instead of being invented.
"""

import io
import zipfile
from typing import Any, Dict, List, Mapping, Optional, Tuple

from spikeforge.serving.safetensor_bundle import STBundle
from spikeforge.serving.st_build import (
    assemble,
    clean_state,
    expected_doc,
    frozen_encode,
    model_doc,
)
from spikeforge.serving.st_format import checksum


def import_v1_bundle(
    path: str, out: Optional[str] = None,
    expected: Optional[Mapping[str, Any]] = None,
) -> Tuple[STBundle, Dict[str, Any]]:
    """Import a v1 ``.spkf`` bundle; report preserved/missing fields."""
    from spikeforge.serving.bundle import DeploymentBundle

    with open(path, "rb") as handle:
        original = handle.read()
    legacy = DeploymentBundle.load(path)
    bundle, missing = _convert(path, original, legacy, expected)
    if out:
        bundle.save(out)
    digest = bundle.manifest["provenance"]["source_sha256"]
    return bundle, {"source_sha256": digest, "missing": missing}


def _convert(
    path: str, original: bytes, legacy: Any,
    expected: Optional[Mapping[str, Any]],
) -> Tuple[STBundle, List[str]]:
    """Convert a loaded v1 bundle into version 2 parts."""
    weights = clean_state(legacy.resolved_weights())
    encode = frozen_encode(legacy.encode_config, {})
    doc = model_doc(legacy.manifest["spec"], legacy.manifest)
    exp_doc, exp_missing = expected_doc(expected)
    record = _v1_record(path, original, legacy)
    missing = exp_missing + ["provenance.dataset"]
    bundle = assemble(weights, encode, dict(legacy.preprocessing), doc,
                      exp_doc, record, missing, legacy.manifest,
                      "spkf-v1", 1, legacy.graph)
    return bundle, missing


def _v1_record(
    path: str, original: bytes, legacy: Any
) -> Dict[str, Any]:
    """Return the provenance record preserving v1 originals."""
    with zipfile.ZipFile(io.BytesIO(original)) as zipf:
        entries = sorted(zipf.namelist())
    return {
        "dataset": None,
        "topology": legacy.manifest.get("topology"),
        "source": f"spkf-v1:{path}",
        "source_sha256": checksum(original),
        "source_entries": entries,
    }
