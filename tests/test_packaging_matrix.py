"""The matrix-row co-installability gate (#98).

The 0.6.0 release shipped a ``compatibility.json`` row that cannot be
installed: core went 0.5.2 -> 0.6.0 while serve/targets/io stayed on the
versions pinning ``spikeforge~=0.5.x``. ``check_coordinated_minor_bump``
refuses that shape -- a core MINOR bump with stale satellites -- and
``release.yml`` additionally dry-run resolves the latest row from local
builds. These tests prove the static half fails on the pre-fix 0.6.0 row
(exactly as recorded in git history) and passes on the corrected one.
"""

import json
from pathlib import Path

from scripts.check_packaging_guards import check_coordinated_minor_bump

_ROOT = Path(__file__).resolve().parent.parent
_MATRIX = _ROOT / "compatibility.json"
_RELEASE = _ROOT / ".github" / "workflows" / "release.yml"

_ROW_052 = {
    "spikeforge": "0.5.2",
    "spikeforge-server": "0.3.4",
    "spikeforge-serve": "0.2.4",
    "spikeforge-clients": "0.1.1",
    "spikeforge-targets": "0.2.1",
    "spikeforge-io": "0.1.4",
    "spikeforge-hub": "0.3.2",
    "dashboard": "0.3.0",
}
# The 0.6.0 row as first shipped: serve/targets/io forgot their bump.
_PREFIX_060 = {
    "spikeforge": "0.6.0",
    "spikeforge-server": "0.4.0",
    "spikeforge-serve": "0.2.4",
    "spikeforge-clients": "0.1.1",
    "spikeforge-targets": "0.2.1",
    "spikeforge-io": "0.1.4",
    "spikeforge-hub": "0.4.0",
    "dashboard": "0.3.0",
}
# The corrected row: pin-only patch releases for the three laggards.
_FIXED_060 = {
    "spikeforge": "0.6.0",
    "spikeforge-server": "0.4.0",
    "spikeforge-serve": "0.2.5",
    "spikeforge-clients": "0.1.1",
    "spikeforge-targets": "0.2.2",
    "spikeforge-io": "0.1.5",
    "spikeforge-hub": "0.4.0",
    "dashboard": "0.3.0",
}


def test_prefix_060_row_is_refused() -> None:
    """The shipped-broken 0.6.0 combination names its stale satellites."""
    problem = check_coordinated_minor_bump([_ROW_052, _PREFIX_060])
    assert problem is not None
    assert "spikeforge-serve" in problem
    assert "spikeforge-targets" in problem
    assert "spikeforge-io" in problem


def test_corrected_060_row_passes() -> None:
    """Coordinated patch releases satisfy the gate; clients may ride."""
    assert check_coordinated_minor_bump([_ROW_052, _FIXED_060]) is None


def test_current_matrix_tail_passes() -> None:
    """The live matrix's last two rows satisfy the gate."""
    with _MATRIX.open("rb") as handle:
        releases = json.load(handle)["releases"]
    assert check_coordinated_minor_bump(releases[-2:]) is None


def test_patch_bump_needs_no_satellite_bump() -> None:
    """A core PATCH bump leaves compatible-release pins valid."""
    current = dict(_ROW_052, spikeforge="0.5.3")
    assert check_coordinated_minor_bump([_ROW_052, current]) is None


def test_unparseable_core_version_fails_closed() -> None:
    """A core version that is not X.Y refuses instead of skipping."""
    current = dict(_FIXED_060, spikeforge="not-a-version")
    problem = check_coordinated_minor_bump([_ROW_052, current])
    assert problem is not None


def test_release_workflow_runs_the_row_gate() -> None:
    """release.yml runs the guards and dry-run resolves the row."""
    text = _RELEASE.read_text(encoding="utf-8")
    assert "scripts/check_packaging_guards.py" in text
    assert "--dry-run" in text
    assert "--find-links" in text
