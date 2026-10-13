"""Assemble the verification report posted to the hub API.

The one binding field is ``passed``: ``spikeforge-hub-api``'s
``hub_api/catalog/trust.py::label_for`` reads
``version.verification["passed"]`` and shows the ``machine-checked`` trust
label only when it is the JSON boolean ``true``. Everything else in this
report (``reasons``, ``checks``, ``summary``) is this runner's proposed
shape for the rest of the contract issue #48 and hub-api issue #4 both
describe in prose ("pass/fail, reasons, NIR/compat/energy summary") but
neither repository had committed code for as of this writing -- see the
pull request description for the coordination note.
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
) -> Dict[str, Any]:
    """Return the JSON-able report body for one verification run.

    ``reasons`` names every failing step in the uploader's own words (see
    :mod:`hub_verify.errors`); it is empty exactly when ``passed`` is True.
    ``checks`` carries the bundle/inspect/compat/drift/energy detail a
    listing page can render even for a passing artifact.
    """
    return {
        "version_id": version_id,
        "passed": bool(passed),
        "reasons": list(reasons),
        "checks": checks,
        "summary": summary,
        "generated_at": _now_iso(),
        "runner": {"workflow": "hub-verify.yml", "run_url": run_url},
    }


def rejection_summary(reasons: List[str]) -> str:
    """Return a one-line summary naming the first rejection reason."""
    if not reasons:
        return "rejected with no named reason (this is itself a bug)"
    return f"rejected: {reasons[0]}"
