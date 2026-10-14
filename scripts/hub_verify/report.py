"""Assemble the verification report posted to the hub API.

The one binding field is ``passed``: ``spikeforge-hub-api``'s
``hub_api/catalog/trust.py::label_for`` reads
``version.verification["passed"]`` and shows the ``machine-checked`` trust
label only when it is the JSON boolean ``true``. The rest of the shape
(``reasons``, ``checks``, ``summary``) is the reconciled contract pinned in
``plans/hub_accounts_plan.md`` §7.3: the hub side stores every other field
verbatim, so this runner owns the detail shape.
"""

import datetime
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
    )


def build_report(
    *,
    version_id: str,
    passed: bool,
    reasons: List[str],
    checks: Dict[str, Optional[Any]],
    summary: str,
    run_url: str,
    framework: Optional[str] = None,
    kind: Optional[str] = None,
    topology: Optional[str] = None,
    input_shape: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the JSON-able report body for one verification run.

    ``reasons`` names every failing step in the uploader's own words (see
    :mod:`hub_verify.errors`); it is empty exactly when ``passed`` is True.
    ``checks`` carries the bundle/inspect/compat/drift/energy detail a
    listing page can render even for a passing artifact. The identity
    fields (``framework``, ``kind``, ``topology``, ``input_shape``) name
    what the artifact *is*, measured by this runner from the bytes -- the
    hub side renders them verbatim, so they are always present and null
    whenever the run failed before the funnel completed.
    """
    return {
        "version_id": version_id,
        "passed": bool(passed),
        "reasons": list(reasons),
        "checks": checks,
        "summary": summary,
        "generated_at": _now_iso(),
        "runner": {"workflow": "hub-verify.yml", "run_url": run_url},
        "framework": framework,
        "kind": kind,
        "topology": topology,
        "input_shape": input_shape,
    }


def rejection_summary(reasons: List[str]) -> str:
    """Return a one-line summary naming the first rejection reason."""
    if not reasons:
        return "rejected with no named reason (this is itself a bug)"
    return f"rejected: {reasons[0]}"
