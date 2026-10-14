"""Schema validators for one governed registry entry.

Split out of :mod:`spikeforge_hub.registry` so the governance layer stays
within the file-length limit. The shared primitives (required-field, id,
and projection helpers) live in :mod:`spikeforge_hub._validation`; the
wrappers here bind the registry's schema constants and error types.
"""

from typing import Any, Dict, Mapping

from spikeforge_hub._validation import (
    check_id,
    check_required,
    project_fields,
)
from spikeforge_hub._validation import label as _label
from spikeforge_hub._validation import text as _text
from spikeforge_hub.registry_errors import (
    RegistrySchemaError,
    RegistryVersionError,
)
from spikeforge_hub.registry_schema import (
    _FIELDS,
    _ID,
    _REQUIRED,
    _SHA256,
    _VERSION,
    COMPATIBILITY_VERDICTS,
    LINEAGE_ORDER,
    REGISTRY_SCHEMA_VERSION,
    STAGES,
    STATUSES,
)


def _check_required(data: Mapping[str, Any]) -> None:
    """Reject an entry that omits or blanks a required text field."""
    check_required(data, _REQUIRED, RegistrySchemaError)


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
    check_id(
        data, _ID, "[A-Za-z0-9][A-Za-z0-9._/-]*", RegistrySchemaError
    )


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
    values = project_fields(data, _FIELDS)
    values["lineage"] = dict(data.get("lineage") or {})
    values["created_at"] = float(data.get("created_at") or 0.0)
    values["schema_version"] = int(
        data.get("schema_version", REGISTRY_SCHEMA_VERSION)
    )
    return values
