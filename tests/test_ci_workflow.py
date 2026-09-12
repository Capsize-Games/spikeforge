"""Text guards on ``.github/workflows/ci.yml``.

Nothing imports the CI workflow, so these assertions read the YAML as text to
keep its load-bearing jobs from silently disappearing. The ``test-deploy`` job
is the only CI run of the simulator-backed parity matrix on a shipped topology
(``fc_legacy``), so it gets an explicit guard: the job must exist, run the
matrix CLI, assert one cell per registered target, and stay dependency-light
(no vendor SDK extra installed — unavailable cells are reported honestly).
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CI = _ROOT / ".github" / "workflows" / "ci.yml"


def _job_block(name: str) -> str:
    """Return the YAML text of the top-level job ``name`` from ci.yml."""
    text = _CI.read_text(encoding="utf-8")
    match = re.search(
        rf"^  {re.escape(name)}:\n(.*?)(?=^  \S|\Z)",
        text,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"CI job {name!r} is missing from ci.yml"
    return match.group(0)


def test_ci_has_a_test_deploy_job() -> None:
    """ci.yml declares the ``test-deploy`` job."""
    assert _job_block("test-deploy")


def test_test_deploy_runs_the_matrix_on_fc_legacy() -> None:
    """The job runs the matrix CLI on the linear ``fc_legacy`` topology."""
    job = _job_block("test-deploy")
    assert "spikeforge.cli.verify test-deploy" in job
    assert "fc_legacy" in job


def test_test_deploy_asserts_one_cell_per_registered_target() -> None:
    """The job asserts the matrix JSON holds one cell per registered target."""
    job = _job_block("test-deploy")
    assert 'payload["ok"] is True' in job
    assert 'payload["cells"]' in job
    assert "registry.target_names()" in job


def test_test_deploy_installs_no_vendor_sdk() -> None:
    """The job stays dependency-light: no vendor SDK extra is installed."""
    job = _job_block("test-deploy")
    for extra in ("norse", "lava", "tonic", "rockpool", "sinabs", "brian"):
        assert f"[{extra}" not in job
    assert "spikeforge-targets" in job
