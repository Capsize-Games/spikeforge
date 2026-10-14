"""The governed registry entry type and its constructors.

Split out of :mod:`spikeforge_hub.registry` so the governance layer stays
within the file-length limit. Parsing validates through
:mod:`spikeforge_hub.registry_checks`; signing, promotion, storage, and
catalog governance live in their own modules.
"""

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from spikeforge_hub._validation import text as _text
from spikeforge_hub.entry import UNVERIFIED_CANDIDATE, HubEntry
from spikeforge_hub.registry_checks import (
    _check_artifact,
    _check_compat,
    _check_id,
    _check_lineage,
    _check_required,
    _check_schema_version,
    _check_stage,
    _check_status,
    _check_version,
    _fields,
)
from spikeforge_hub.registry_errors import (
    RegistryIntegrityError,
    RegistrySchemaError,
)
from spikeforge_hub.registry_schema import (
    _FIELDS,
    LINEAGE_ORDER,
    REGISTRY_SCHEMA_VERSION,
    STAGES,
)
from spikeforge_hub.verify import VERIFIED, verify_file


def _checkpoint_artifact(name: str) -> Tuple[str, Optional[str]]:
    """Return the checkpoint's stored path and sha256 (None if missing)."""
    from spikeforge.network import model_store

    path = model_store.path_for(name)
    if not os.path.exists(path):
        return path, None
    from spikeforge_hub.verify import file_sha256

    return path, file_sha256(path)


def _checkpoint_links(
    name: str, lineage: Optional[Mapping[str, Optional[str]]]
) -> Dict[str, Optional[str]]:
    """Return lineage links with the model link defaulted to ``name``."""
    links = dict(lineage or {})
    links.setdefault("model", name)
    return links


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
        """Build an entry for a saved checkpoint, recording its sha256."""
        path, digest = _checkpoint_artifact(name)
        links = _checkpoint_links(name, lineage)
        return cls(id=entry_id or name, name=name, stage=stage,
                   status=status, version=version, artifact_path=path,
                   artifact_sha256=digest, compat=compat, lineage=links,
                   **extra)

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
