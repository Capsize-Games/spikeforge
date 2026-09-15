"""Governed, signable registry of promoted artifacts and their lineage.

The model hub already validates a *catalog* of curated artifacts
(:mod:`spikeforge_hub.catalog`) and classifies an artifact against a shipped
preset (:mod:`spikeforge_hub.compat`). This module adds the missing
**governance** layer the PT-W7 workstream calls for: an explicit promotion
lifecycle (``dev -> staging -> prod``), a recorded approver, an artifact
signature that can be verified, and lineage from dataset to deployment.

Design choices (documented because the plan leaves them open):

* **Extend the hub, not a new distribution.** The plan offers
  ``spikeforge-registry`` as *new (or extend hub)*; the governance layer lives
  in ``spikeforge_hub`` and owns the ``spikeforge-registry`` console script, so
  no third distribution (and no extra import root) is introduced.
* **Reuse the existing machinery.** Compatibility reuses the
  :data:`~spikeforge_hub.compat.EXACT`/:data:`~spikeforge_hub.compat.MAPPABLE`
  vocabulary, and checksum/verification reuses
  :func:`spikeforge_hub.verify.verify_file`, so there is one definition of
  "verified" and "compatible".
* **Signing is dependency-free.** A signature is an HMAC-SHA256 over the
  canonical (sorted-key) JSON of the entry's unsigned projection. It detects a
  tampered or absent signature without pulling in a crypto stack; a caller that
  needs asymmetric signing can swap :func:`sign_entry` for its own.
"""

import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from spikeforge.config import REGISTRY_DIR
from spikeforge_hub.compat import EXACT, INCOMPATIBLE, MAPPABLE
from spikeforge_hub.entry import UNVERIFIED_CANDIDATE, HubEntry
from spikeforge_hub.errors import HubCatalogError
from spikeforge_hub.registry_errors import (
    RegistryApprovalError,
    RegistryCompatibilityError,
    RegistryError,
    RegistryIntegrityError,
    RegistryLifecycleError,
    RegistryPromotionError,
    RegistrySchemaError,
    RegistrySignatureError,
    RegistryVersionError,
)
from spikeforge_hub.verify import VERIFIED, verify_file

#: Bumped whenever the registry document schema changes incompatibly.
REGISTRY_SCHEMA_VERSION = 1

#: Promotion stages, in the only order a promotion may advance through.
STAGES: Tuple[str, ...] = ("dev", "staging", "prod")

#: Lifecycle statuses an entry may carry. ``active`` may be promoted; the other
#: two are terminal for promotion, and ``deprecated`` is additionally reported.
STATUSES: Tuple[str, ...] = ("active", "deprecated", "retired")

#: Lineage links, in the order the plan names them (dataset -> ... deployment).
LINEAGE_ORDER: Tuple[str, ...] = ("dataset", "model", "bundle", "deployment")

#: Compatibility verdicts the hub can record; only the first two may ship.
COMPATIBILITY_VERDICTS: Tuple[str, ...] = (EXACT, MAPPABLE, INCOMPATIBLE)

#: Default registry document name under :data:`spikeforge.config.REGISTRY_DIR`.
REGISTRY_FILE = "registry.json"

#: An entry id that cannot escape a directory or confuse the cache layout.
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
#: A concrete dotted version, e.g. ``0.1`` or ``0.1.0``.
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+){0,2}$")
#: A hex sha256 digest.
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED: Tuple[str, ...] = ("id", "name", "stage", "status", "version")
_FIELDS: Tuple[str, ...] = _REQUIRED + (
    "artifact_path",
    "artifact_sha256",
    "compat",
    "approver",
    "approved_at",
    "signature",
    "lineage",
    "created_at",
    "schema_version",
)


def _text(value: Any) -> bool:
    """Return True when ``value`` is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


def _label(data: Mapping[str, Any]) -> str:
    """Return the entry id for messages, falling back to ``(unnamed)``."""
    value = data.get("id")
    return value if isinstance(value, str) and value else "(unnamed)"


def _checkpoint_artifact(name: str) -> Tuple[str, Optional[str]]:
    """Return the checkpoint's stored path and sha256 (None if missing)."""
    from spikeforge.network import model_store

    path = model_store.path_for(name)
    if not os.path.exists(path):
        return path, None
    from spikeforge_hub.verify import file_sha256

    return path, file_sha256(path)


def stage_rank(stage: str) -> int:
    """Return the ordinal of ``stage``, or raise a typed schema error."""
    try:
        return STAGES.index(stage)
    except ValueError:
        raise RegistrySchemaError(
            stage or "(unnamed)",
            f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}",
        ) from None


@dataclass(frozen=True)
class RegistryEntry:
    """One governed artifact: location, stage, approval, and lineage.

    ``stage`` and ``status`` are the governance state; ``artifact_path`` and
    ``artifact_sha256`` describe the bytes; ``compat`` records the hub
    compatibility verdict a promotion to ``prod`` relies on; ``approver`` and
    ``approved_at`` are the audit trail; ``signature`` is the HMAC over the
    unsigned projection; and ``lineage`` walks dataset -> deployment.
    """

    id: str
    name: str
    stage: str = "dev"
    status: str = "active"
    version: str = "0.1.0"
    artifact_path: Optional[str] = None
    artifact_sha256: Optional[str] = None
    compat: Optional[str] = None
    approver: Optional[str] = None
    approved_at: Optional[float] = None
    signature: Optional[str] = None
    lineage: Mapping[str, Optional[str]] = field(default_factory=dict)
    created_at: float = 0.0
    schema_version: int = REGISTRY_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegistryEntry":
        """Validate ``data`` and build an entry, raising on any problem."""
        if not isinstance(data, Mapping):
            raise RegistrySchemaError("(unnamed)", "entry must be a mapping")
        _check_required(data)
        _check_schema_version(data)
        _check_id(data)
        _check_stage(data)
        _check_status(data)
        _check_version(data)
        _check_artifact(data)
        _check_compat(data)
        _check_lineage(data)
        return cls(**_fields(data))

    @classmethod
    def from_checkpoint(
        cls,
        name: str,
        *,
        entry_id: Optional[str] = None,
        stage: str = "dev",
        status: str = "active",
        version: str = "0.1.0",
        compat: Optional[str] = None,
        lineage: Optional[Mapping[str, Optional[str]]] = None,
        **extra: Any,
    ) -> "RegistryEntry":
        """Build an entry for a saved checkpoint, recording its sha256.

        The id defaults to ``name`` and the model lineage link is filled
        from ``name`` unless overridden.
        """
        path, digest = _checkpoint_artifact(name)
        links = dict(lineage or {})
        links.setdefault("model", name)
        return cls(
            id=entry_id or name,
            name=name,
            stage=stage,
            status=status,
            version=version,
            artifact_path=path,
            artifact_sha256=digest,
            compat=compat,
            lineage=links,
            **extra,
        )

    @classmethod
    def from_catalog_entry(
        cls, entry: HubEntry, *, stage: str = "dev", status: str = "active",
        version: str = "0.1.0", **extra: Any,
    ) -> "RegistryEntry":
        """Adapt a curated :class:`HubEntry` into a governed registry entry.

        An unverified candidate maps to ``deprecated`` status, so it lists
        but never promotes into ``prod`` without a concrete license.
        """
        unverified = entry.license == UNVERIFIED_CANDIDATE
        resolved = "deprecated" if unverified else status
        return cls(
            id=entry.id, name=entry.name, stage=stage, status=resolved,
            version=version, artifact_sha256=entry.sha256,
            lineage={"dataset": entry.framework, "model": entry.id},
            **extra,
        )

    def unsigned(self) -> Dict[str, Any]:
        """Return the canonical projection signed/verified as this entry."""
        values = asdict(self)
        values.pop("signature", None)
        return values

    def validated(self) -> "RegistryEntry":
        """Return this entry after re-running every schema check."""
        return self.from_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        """Return the entry as a JSON-able mapping in schema field order."""
        values = asdict(self)
        return {name: values[name] for name in _FIELDS}


def _check_required(data: Mapping[str, Any]) -> None:
    """Reject an entry that omits or blanks a required text field."""
    missing = [key for key in _REQUIRED if not _text(data.get(key))]
    if missing:
        raise RegistrySchemaError(
            _label(data), f"missing required fields: {', '.join(missing)}"
        )


def _check_schema_version(data: Mapping[str, Any]) -> None:
    """Reject an entry pinned to an unsupported registry schema."""
    value = data.get("schema_version", REGISTRY_SCHEMA_VERSION)
    if int(value) != REGISTRY_SCHEMA_VERSION:
        raise RegistryVersionError(
            _label(data),
            f"schema {value!r} is not {REGISTRY_SCHEMA_VERSION}",
        )


def _check_id(data: Mapping[str, Any]) -> None:
    """Reject an id that could escape or confuse the registry layout."""
    entry_id = str(data.get("id", ""))
    if not _ID.match(entry_id):
        raise RegistrySchemaError(
            _label(data), "id must match [A-Za-z0-9][A-Za-z0-9._/-]*"
        )
    if ".." in entry_id:
        raise RegistrySchemaError(_label(data), "id must not contain '..'")


def _check_stage(data: Mapping[str, Any]) -> None:
    """Reject an unknown promotion stage."""
    if data.get("stage") not in STAGES:
        raise RegistrySchemaError(
            _label(data),
            f"unknown stage {data.get('stage')!r}; expected one of "
            f"{', '.join(STAGES)}",
        )


def _check_status(data: Mapping[str, Any]) -> None:
    """Reject an unknown lifecycle status."""
    if data.get("status") not in STATUSES:
        raise RegistrySchemaError(
            _label(data),
            f"unknown status {data.get('status')!r}; expected one of "
            f"{', '.join(STATUSES)}",
        )


def _check_version(data: Mapping[str, Any]) -> None:
    """Reject a version that is not a concrete dotted number."""
    version = data.get("version")
    if not _text(version) or not _VERSION.match(str(version)):
        raise RegistryVersionError(
            _label(data), f"version {version!r} must be major[.minor[.patch]]"
        )


def _check_artifact(data: Mapping[str, Any]) -> None:
    """Reject a malformed recorded checksum."""
    digest = data.get("artifact_sha256")
    if digest is None:
        return
    if not isinstance(digest, str) or not _SHA256.match(digest):
        raise RegistrySchemaError(
            _label(data), "artifact_sha256 must be a 64-character hex digest"
        )


def _check_compat(data: Mapping[str, Any]) -> None:
    """Reject a compatibility verdict outside the hub vocabulary."""
    verdict = data.get("compat")
    if verdict is not None and verdict not in COMPATIBILITY_VERDICTS:
        raise RegistrySchemaError(
            _label(data),
            f"unknown compat {verdict!r}; expected one of "
            f"{', '.join(COMPATIBILITY_VERDICTS)}",
        )


def _check_lineage(data: Mapping[str, Any]) -> None:
    """Reject a lineage key the plan does not name."""
    lineage = data.get("lineage") or {}
    if not isinstance(lineage, Mapping):
        raise RegistrySchemaError(_label(data), "lineage must be a mapping")
    unknown = sorted(set(lineage) - set(LINEAGE_ORDER))
    if unknown:
        raise RegistrySchemaError(
            _label(data), f"unknown lineage links: {', '.join(unknown)}"
        )


def _fields(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Project ``data`` onto the declared schema fields."""
    values = {name: data.get(name) for name in _FIELDS}
    values["lineage"] = dict(data.get("lineage") or {})
    values["created_at"] = float(data.get("created_at") or 0.0)
    values["schema_version"] = int(
        data.get("schema_version", REGISTRY_SCHEMA_VERSION)
    )
    return values


def canonical(entry: RegistryEntry) -> str:
    """Return the canonical JSON signed/verified for ``entry``."""
    payload = entry.unsigned()
    payload.pop("signature", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sign_entry(entry: RegistryEntry, key: Any) -> str:
    """Return the hex HMAC-SHA256 signature of ``entry`` under ``key``."""
    secret = _as_key(key)
    digest = hmac.new(secret, canonical(entry).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def verify_entry(entry: RegistryEntry, key: Any) -> bool:
    """Return True when ``entry`` carries a matching signature.

    Raises :class:`RegistrySignatureError` when the signature is absent or does
    not match, so a caller that needs a boolean can catch the typed error.
    """
    if not entry.signature:
        raise RegistrySignatureError(entry.id, "entry carries no signature")
    expected = sign_entry(entry, key)
    if not hmac.compare_digest(entry.signature, expected):
        raise RegistrySignatureError(
            entry.id, "signature does not match the entry", expected
        )
    return True


def _as_key(key: Any) -> bytes:
    """Return ``key`` as bytes, refusing an empty secret."""
    if isinstance(key, bytes):
        secret = key
    elif isinstance(key, str):
        secret = key.encode("utf-8")
    else:
        raise RegistrySignatureError(
            "(key)", "signing key must be str or bytes"
        )
    if not secret:
        raise RegistrySignatureError("(key)", "signing key must not be empty")
    return secret


def lineage_report(entry: RegistryEntry) -> Dict[str, Any]:
    """Return the ordered dataset -> deployment chain and its completeness."""
    chain: List[Dict[str, str]] = []
    missing: List[str] = []
    for link in LINEAGE_ORDER:
        value = (entry.lineage or {}).get(link)
        if _text(value):
            chain.append({"link": link, "ref": str(value)})
        else:
            missing.append(link)
    return {
        "chain": chain,
        "complete": not missing,
        "missing": missing,
    }


def verify_artifact(entry: RegistryEntry) -> Dict[str, Any]:
    """Return the artifact's verification report, or ``unverified`` if absent.

    A recorded checksum that does not match the bytes on disk raises
    :class:`RegistryIntegrityError`; a matching checksum reports ``verified``;
    an entry without a checksum reports ``unverified`` (never a silent pass),
    mirroring :mod:`spikeforge_hub.verify`.
    """
    if not entry.artifact_path or not entry.artifact_sha256:
        return {"status": "unverified", "reason": "no recorded checksum"}
    if not os.path.exists(entry.artifact_path):
        raise RegistryIntegrityError(
            entry.id, f"artifact not found: {entry.artifact_path}"
        )
    result = verify_file(entry.artifact_path, sha256=entry.artifact_sha256)
    if result.status != VERIFIED:
        raise RegistryIntegrityError(
            entry.id, result.reason or "artifact does not match its checksum"
        )
    return {"status": result.status, "reason": result.reason}


def _check_stage_transition(entry: RegistryEntry, to_stage: str) -> None:
    """Raise unless ``to_stage`` is the entry's next lifecycle stage."""
    current = stage_rank(entry.stage)
    target = stage_rank(to_stage)
    if target != current + 1:
        raise RegistryPromotionError(
            entry.id,
            f"cannot move {entry.stage!r} -> {to_stage!r}; expected "
            f"{STAGES[current + 1]!r}",
        )
    if entry.status != "active":
        raise RegistryLifecycleError(
            entry.id,
            entry.status,
            f"only 'active' may be promoted to {to_stage!r}",
        )


def _check_approval_and_compat(
    entry: RegistryEntry, to_stage: str, approver: str
) -> None:
    """Raise unless ``approver`` is named and prod compat is satisfied."""
    if not _text(approver):
        raise RegistryApprovalError(entry.id, "a named approver is required")
    if to_stage == "prod" and entry.compat not in (EXACT, MAPPABLE):
        raise RegistryCompatibilityError(
            entry.id,
            entry.compat or "(none)",
            "a 'prod' promotion requires an exact/mappable verdict",
        )


def _validate_promotion(
    entry: RegistryEntry, to_stage: str, approver: str, check_artifact: bool
) -> None:
    """Raise a typed error when ``entry`` cannot be promoted to a stage."""
    _check_stage_transition(entry, to_stage)
    _check_approval_and_compat(entry, to_stage, approver)
    if check_artifact:
        verify_artifact(entry)


def _promoted_entry(
    entry: RegistryEntry, to_stage: str, approver: str,
    at: Optional[float], key: Any,
) -> RegistryEntry:
    """Return ``entry`` advanced to ``to_stage`` and freshly re-signed."""
    promoted = replace(
        entry,
        stage=to_stage,
        approver=approver.strip(),
        approved_at=float(at if at is not None else time.time()),
        signature=None,
    )
    return replace(promoted, signature=sign_entry(promoted, key))


def promote(
    entry: RegistryEntry,
    to_stage: str,
    approver: str,
    key: Any,
    *,
    at: Optional[float] = None,
    check_artifact: bool = True,
) -> RegistryEntry:
    """Return ``entry`` advanced to ``to_stage`` with a recorded approval.

    Refused with a typed error on a skipped/terminal/missing-approval/
    incompatible/checksum-failed transition; on success the entry is
    re-signed so the signature covers the new stage.
    """
    entry.validated()
    _validate_promotion(entry, to_stage, approver, check_artifact)
    return _promoted_entry(entry, to_stage, approver, at, key)


def default_registry_path() -> str:
    """Return the default registry document path under ``REGISTRY_DIR``."""
    return os.path.join(REGISTRY_DIR, REGISTRY_FILE)


def _read_json_file(path: str) -> Any:
    """Return the parsed JSON at ``path``, wrapping I/O and parse errors."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise RegistrySchemaError(
            "(registry)", f"cannot read {path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RegistrySchemaError(
            "(registry)", f"malformed JSON: {exc}"
        ) from exc


def _check_registry_version(data: Mapping[str, Any]) -> int:
    """Return the document's schema version, or raise when unsupported."""
    version = int(data.get("version", REGISTRY_SCHEMA_VERSION))
    if version != REGISTRY_SCHEMA_VERSION:
        raise RegistryVersionError(
            "(registry)",
            f"document version {version!r} is not "
            f"{REGISTRY_SCHEMA_VERSION}",
        )
    return version


def _parse_registry_entries(
    raw: Any,
) -> Tuple[Tuple[RegistryEntry, ...], Tuple[str, ...]]:
    """Parse each raw entry, collecting a message for every failure."""
    if not isinstance(raw, list):
        raise RegistrySchemaError("(registry)", "'entries' must be a list")
    found: List[RegistryEntry] = []
    problems: List[str] = []
    for index, item in enumerate(raw):
        try:
            found.append(RegistryEntry.from_dict(item))
        except RegistryError as error:
            problems.append(_position(index, item, error))
    return tuple(found), tuple(problems)


@dataclass(frozen=True)
class Registry:
    """A collection of governed entries with deterministic ordering.

    Loading validates every entry and collects each problem instead of
    dropping it, so :meth:`issues` names everything wrong with a document while
    :meth:`entries` returns only the valid ones — the same honest contract the
    catalog uses.
    """

    entries: Tuple[RegistryEntry, ...] = ()
    issues: Tuple[str, ...] = ()
    version: int = REGISTRY_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Registry":
        """Validate a registry document, collecting every entry issue."""
        if not isinstance(data, Mapping):
            raise RegistrySchemaError("(registry)", "root must be a mapping")
        _check_registry_version(data)
        found, problems = _parse_registry_entries(data.get("entries", []))
        return cls(entries=found, issues=problems)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Registry":
        """Load the registry at ``path`` (or the default path).

        A missing file yields an empty, valid registry so a first promotion
        does not need a pre-seeded document.
        """
        resolved = path or default_registry_path()
        if not os.path.exists(resolved):
            return cls()
        return cls.from_dict(_read_json_file(resolved))

    def save(self, path: Optional[str] = None) -> str:
        """Write the registry to ``path`` and return that path."""
        resolved = path or default_registry_path()
        directory = os.path.dirname(resolved)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.to_dict(), indent=2) + "\n")
        return resolved

    def ordered(self) -> Tuple[RegistryEntry, ...]:
        """Return entries ordered deterministically by stage, id, version."""
        return tuple(
            sorted(
                self.entries,
                key=lambda item: (
                    stage_rank(item.stage),
                    item.id,
                    item.version,
                ),
            )
        )

    def get(self, entry_id: str) -> Optional[RegistryEntry]:
        """Return ``entry_id``'s entry at its furthest stage, or None."""
        matches = [item for item in self.entries if item.id == entry_id]
        if not matches:
            return None
        return max(matches, key=lambda item: stage_rank(item.stage))

    def add(self, entry: RegistryEntry) -> "Registry":
        """Return a registry with ``entry`` replacing a same-stage peer."""
        entry.validated()
        kept = tuple(
            item
            for item in self.entries
            if not (item.id == entry.id and item.stage == entry.stage)
        )
        return replace(self, entries=kept + (entry,))

    def to_dict(self) -> Dict[str, Any]:
        """Return the registry as a JSON-able mapping in schema order."""
        return {
            "version": int(self.version),
            "entries": [item.to_dict() for item in self.ordered()],
        }

    def flagged(self) -> List[Dict[str, Any]]:
        """Return the deprecated/retired entries that need attention."""
        return [
            {"id": item.id, "stage": item.stage, "status": item.status}
            for item in self.ordered()
            if item.status != "active"
        ]


def _position(index: int, item: Any, error: RegistryError) -> str:
    """Return a positional prefix for an entry that failed validation."""
    label = _label(item) if isinstance(item, Mapping) else f"#{index}"
    return f"{label}: {error.detail}"


def _unverified_issues(entries: List[HubEntry]) -> List[str]:
    """Return one governance issue per unverified-candidate entry."""
    return [
        f"{entry.id}: unverified candidate; lifecycle maps to "
        "'deprecated' and it may not be promoted to 'prod'"
        for entry in entries
        if entry.license == UNVERIFIED_CANDIDATE
    ]


def validate_catalog(
    path: Optional[str] = None,
) -> Tuple[List[HubEntry], List[str]]:
    """Validate the catalog at ``path`` and adapt it to governance.

    Returns the valid catalog entries plus every problem, so a governance
    command can report catalog schema/version issues and lifecycle status in
    one pass without re-implementing the catalog loader.
    """
    # ``spikeforge_hub.catalog`` is shadowed at package level by the
    # ``catalog()`` function the public API exports, so the submodule must be
    # resolved through the import system rather than attribute access.
    hub_catalog = import_module("spikeforge_hub.catalog")

    target = Path(path) if path else hub_catalog.CATALOG_PATH
    try:
        entries, issues = hub_catalog.load_catalog(target)
    except HubCatalogError as error:
        return [], [str(error)]
    return entries, issues + _unverified_issues(entries)


def catalog_governance() -> List[Dict[str, Any]]:
    """Return the governed view of the bundled catalog, deterministically."""
    hub_catalog = import_module("spikeforge_hub.catalog")

    governed = [
        RegistryEntry.from_catalog_entry(entry)
        for entry in hub_catalog.entries()
    ]
    return [
        {
            **item.to_dict(),
            "lineage": lineage_report(item),
        }
        for item in sorted(governed, key=lambda item: item.id)
    ]


#: Re-exported so callers can build a stage list without importing STAGES.
StageList = Sequence[str]
