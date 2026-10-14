"""Promotion of a registry entry through its lifecycle stages.

Split out of :mod:`spikeforge_hub.registry` so the governance layer stays
within the file-length limit. A promotion advances exactly one stage, names
an approver, gates ``prod`` on compatibility, and re-signs the entry.
"""

import time
from dataclasses import replace
from typing import Any, Optional

from spikeforge_hub._validation import text as _text
from spikeforge_hub.compat import EXACT, MAPPABLE
from spikeforge_hub.registry_entry import (
    RegistryEntry,
    stage_rank,
    verify_artifact,
)
from spikeforge_hub.registry_errors import (
    RegistryApprovalError,
    RegistryCompatibilityError,
    RegistryLifecycleError,
    RegistryPromotionError,
)
from spikeforge_hub.registry_schema import STAGES
from spikeforge_hub.registry_signing import sign_entry


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
