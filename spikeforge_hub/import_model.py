"""Orchestrate the inspect -> compat -> promote funnel for one artifact.

Gate three. An artifact is inspected, classified against the shipped presets,
and only when the verdict is ``exact``/``mappable`` are its weights mapped onto
the preset, a drift check run, and the result saved into ``MODEL_DIR`` with hub
provenance in ``meta``. An incompatible artifact is never promoted; the report
names every mismatch and, for a NIR graph, states that it stays ingestable by
the reference interpreter.

Before any of that, a fetched artifact has to pass the integrity gate: bytes
that fail their published checksum are refused outright, and checksum-less
bytes are refused unless the operator explicitly opts in. The verified flag
is a gate, not advisory text.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from spikeforge_hub import catalog, verify
from spikeforge_hub.compat import CompatibilityVerdict, classify
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import (
    HubArtifactError,
    HubImportError,
    HubIntegrityError,
)
from spikeforge_hub.inspect import (
    NIR_GRAPH,
    STATE_DICT,
    ArtifactReport,
    inspect_artifact,
    read_state_dict,
    resolve_path,
)
from spikeforge_hub.weight_map import load_weights

#: Deterministic probe shape for the promotion drift check.
STEPS = 4
BATCH = 1
FEATURES = 28 * 28
SEED = 0

#: Operator opt-in, for callers with no CLI flag (notably the socket path):
#: when set to a truthy value, checksum-less entries import without the
#: explicit ``allow_unverified`` argument. Failed checksums are still refused.
ALLOW_UNVERIFIED_ENV = "SPIKEFORGE_HUB_ALLOW_UNVERIFIED"

#: Sources whose bytes are never fetched, so there is nothing to verify.
#: Bundled presets are rendered locally from this install's own registry.
_LOCAL_SOURCES = ("bundled",)


def _resolve(
    entry_id: Optional[str], path: Optional[str]
) -> Tuple[str, Optional[HubEntry], str]:
    """Return ``(label, entry, artifact_path)`` for the requested artifact."""
    if path is not None:
        if not os.path.exists(path):
            raise HubArtifactError(path, "artifact not found")
        return os.path.basename(path), None, path
    entry = catalog.get(entry_id or "")
    if entry is None:
        raise HubArtifactError(entry_id or "", "unknown catalog entry")
    return entry.id, entry, resolve_path(entry)


def _probe_spikes(spec: Any) -> torch.Tensor:
    """Return a deterministic spike volume shaped for ``spec``."""
    from spikeforge.simulator import input_shape

    torch.manual_seed(SEED)
    flat = torch.rand(STEPS, BATCH, FEATURES)
    return input_shape.to_input_shape(flat, spec)


def _env_allows_unverified() -> bool:
    """Return True when the operator opted in via the environment."""
    return os.environ.get(ALLOW_UNVERIFIED_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _local_verification(reason: str) -> verify.VerificationResult:
    """Return the honest verdict for bytes that were never fetched."""
    return verify.VerificationResult(verify.UNVERIFIED, reason, None, 0)


def _verification(
    entry: Optional[HubEntry], artifact_path: str
) -> verify.VerificationResult:
    """Verify the resolved artifact against its entry's expectations.

    Local bytes -- an explicit path or a locally rendered bundled preset --
    are reported unverified but never gated: the operator already holds
    them. Fetched bytes are checked against the published checksum and
    size, and a directory snapshot against its summed size.
    """
    if entry is None:
        return _local_verification(
            "local file; no published checksum to check"
        )
    if entry.source in _LOCAL_SOURCES:
        return _local_verification(
            "bundled preset rendered locally; nothing was fetched"
        )
    if os.path.isdir(artifact_path):
        return verify.verify_dir(artifact_path, entry.size_bytes)
    return verify.verify_file(
        artifact_path, entry.sha256, entry.size_bytes
    )


def _verification_dict(
    result: verify.VerificationResult,
) -> Dict[str, Any]:
    """Return the JSON-able form of a verification outcome."""
    return {
        "status": result.status,
        "verified": result.verified,
        "reason": result.reason,
        "sha256": result.sha256,
    }


def _gate(
    label: str,
    entry: Optional[HubEntry],
    result: verify.VerificationResult,
    allow_unverified: bool,
) -> None:
    """Refuse fetched bytes that fail verification, before any load.

    A checksum or size mismatch is refused unconditionally. Checksum-less
    bytes are refused unless the operator opted in explicitly, either
    with ``allow_unverified`` or the environment variable. Local bytes
    always pass: they are reported, not gated.
    """
    if result.failed:
        raise HubIntegrityError(label, result.reason or "verification failed")
    if result.verified:
        return
    if entry is None or entry.source in _LOCAL_SOURCES:
        return
    if allow_unverified or _env_allows_unverified():
        return
    raise HubIntegrityError(
        label,
        f"{result.reason}; re-run with --allow-unverified or set "
        f"{ALLOW_UNVERIFIED_ENV}=1 to import these bytes anyway",
    )


def _meta(
    entry: Optional[HubEntry],
    report: ArtifactReport,
    verdict: CompatibilityVerdict,
    verification: verify.VerificationResult,
) -> Dict[str, Any]:
    """Return the checkpoint provenance recorded for a promoted artifact."""
    return {
        "source": "hub",
        "hub_id": entry.id if entry is not None else None,
        "hub_kind": report.kind,
        "hub_verdict": verdict.verdict,
        "topology": verdict.topology,
        "input_shape": entry.input_shape if entry is not None else None,
        "sha256": entry.sha256 if entry is not None else None,
        "verification": _verification_dict(verification),
    }


def _checkpoint_name(label: str) -> str:
    """Return a safe checkpoint name for a promoted artifact."""
    return "hub_" + label.replace("/", "_").replace(os.sep, "_")


def _sources(
    report: ArtifactReport, artifact_path: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Any]]:
    """Return the ``(state_dict, graph)`` weight source for an artifact."""
    if report.kind == STATE_DICT:
        return read_state_dict(artifact_path), None
    from spikeforge.nir_bridge import load_graph

    return None, load_graph(artifact_path)


def _validate(spec: Any, module: Any, graph: Optional[Any]) -> Dict[str, Any]:
    """Run the independent drift check for the promoted module."""
    from spikeforge.nir_bridge import validate

    report = validate(spec, module, _probe_spikes(spec), graph=graph)
    return dict(report)


def _save(label: str, module: Any, meta: Dict[str, Any]) -> str:
    """Save ``module`` into the model store and return its path."""
    from spikeforge.network import model_store

    try:
        return model_store.save(_checkpoint_name(label), module, meta)
    except OSError as exc:
        raise HubImportError(label, f"cannot save checkpoint: {exc}") from exc


def _fail(
    weights: Dict[str, Any], validation: Optional[Dict[str, Any]], reason: str
) -> Dict[str, Any]:
    """Return a not-promoted promotion outcome naming the reason."""
    return {
        "promoted": False,
        "destination": None,
        "weights": weights,
        "validation": validation,
        "reason": reason,
    }


def _success(
    weights: Dict[str, Any], validation: Dict[str, Any], destination: str
) -> Dict[str, Any]:
    """Return a promoted promotion outcome with its destination."""
    return {
        "promoted": True,
        "destination": destination,
        "weights": weights,
        "validation": validation,
        "reason": None,
    }


def _load(
    module: Any, report: ArtifactReport, artifact_path: str
) -> Tuple[Optional[Any], Any]:
    """Return the artifact's ``(graph, weight-load result)`` pair."""
    state, graph = _sources(report, artifact_path)
    return graph, load_weights(module, state_dict=state, graph=graph)


def _final(
    label: str,
    module: Any,
    meta: Dict[str, Any],
    loaded: Any,
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    """Save the validated module and return the promoted outcome."""
    destination = _save(label, module, meta)
    return _success(loaded.to_dict(), validation, destination)


def _promote(
    entry: Optional[HubEntry],
    report: ArtifactReport,
    verdict: CompatibilityVerdict,
    artifact_path: str,
    label: str,
    verification: verify.VerificationResult,
) -> Dict[str, Any]:
    """Build the preset, map weights, validate, and save into the store."""
    from spikeforge.topology.registry import build_topology
    # Built with the artifact's own declared class count, so the weights it
    # carries have somewhere to land; see `expected_state`.
    params = (
        {} if report.num_classes is None
        else {"num_classes": int(report.num_classes)}
    )
    spec, module = build_topology(verdict.topology, params)
    graph, loaded = _load(module, report, artifact_path)
    if not loaded.loaded:
        return _fail(loaded.to_dict(), None, "weights refused: " + loaded.note)
    validation = _validate(spec, module, graph)
    if not validation["within_tolerance"]:
        reason = "drift check failed; refusing to promote"
        return _fail(loaded.to_dict(), validation, reason)
    meta = _meta(entry, report, verdict, verification)
    return _final(label, module, meta, loaded, validation)


def _base(
    label: str,
    report: ArtifactReport,
    verdict: CompatibilityVerdict,
    artifact_path: str,
) -> Dict[str, Any]:
    """Return the shared import report skeleton."""
    return {
        "id": label,
        "kind": report.kind,
        "path": artifact_path,
        "verdict": verdict.to_dict(),
    }


def _skeleton(
    label: str,
    entry: Optional[HubEntry],
    report: ArtifactReport,
    verdict: CompatibilityVerdict,
    artifact_path: str,
    verification: verify.VerificationResult,
) -> Dict[str, Any]:
    """Return the report skeleton with its additive default keys."""
    result = _base(label, report, verdict, artifact_path)
    result["meta"] = _meta(entry, report, verdict, verification)
    result.update({
        "promoted": False,
        "destination": None,
        "weights": None,
        "validation": None,
        "reason": None,
        "nir_ingestable": report.kind == NIR_GRAPH,
        "verification": _verification_dict(verification),
    })
    return result


def _reject(
    report: ArtifactReport, verdict: CompatibilityVerdict, promote: bool
) -> List[str]:
    """Return the honest notes for an artifact that is not promoted."""
    notes = list(verdict.notes)
    if not verdict.compatible:
        notes.append(f"rejected: verdict {verdict.verdict!r}")
        if report.kind == NIR_GRAPH:
            notes.append("still NIR-ingestable by the reference interpreter")
    elif not promote:
        notes.append("promotion skipped; artifact is compatible")
    return notes


def import_model(
    entry_id: Optional[str] = None,
    path: Optional[str] = None,
    topology: Optional[str] = None,
    promote: bool = True,
    allow_unverified: bool = False,
) -> Dict[str, Any]:
    """Run the three-gate funnel and return its JSON-able report.

    Fetched bytes are verified before anything is loaded: a checksum
    mismatch raises :class:`HubIntegrityError` unconditionally, and
    checksum-less bytes raise unless ``allow_unverified`` (or the
    ``SPIKEFORGE_HUB_ALLOW_UNVERIFIED`` environment variable) opts in.
    """
    label, entry, artifact_path = _resolve(entry_id, path)
    verification = _verification(entry, artifact_path)
    _gate(label, entry, verification, allow_unverified)
    report = inspect_artifact(artifact_path)
    verdict = classify(report, topology)
    result = _skeleton(
        label, entry, report, verdict, artifact_path, verification
    )
    result["notes"] = _reject(report, verdict, promote)
    if not verdict.compatible or not promote:
        return result
    result.update(
        _promote(entry, report, verdict, artifact_path, label, verification)
    )
    return result
