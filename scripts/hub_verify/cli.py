"""Fetch, verify, report, and post one dispatched artifact.

Invoked by ``.github/workflows/hub-verify.yml`` as
``python -m hub_verify.cli``. Every input is an environment variable the
workflow sets from the ``repository_dispatch`` payload and this
repository's own secrets/variables -- see the workflow file for exactly
which ones, and for why the callback base URL is a repository-configured
constant rather than something read out of the dispatch payload (a fixed
destination this job trusts, not attacker-influenced data).
"""

import os
import sys
import tempfile
from typing import Dict, List, Tuple

from hub_verify.callback import post_report
from hub_verify.energy import energy_reports
from hub_verify.errors import CallbackError, VerificationStepError
from hub_verify.fetch import fetch_artifact
from hub_verify.funnel import run_funnel
from hub_verify.pipeline import load_and_build
from hub_verify.report import build_report, rejection_summary

_EMPTY_CHECKS: Dict[str, object] = {
    "bundle": None,
    "inspect": None,
    "compat": None,
    "drift": None,
    "energy": None,
}


def _env(name: str) -> str:
    """Return the required environment variable ``name`` or exit loudly."""
    value = os.environ.get(name, "")
    if not value:
        sys.exit(f"hub_verify: missing required env var {name}")
    return value


def _run_url() -> str:
    """Return this job's run URL for the report, or an empty string."""
    server = os.environ.get("GITHUB_SERVER_URL", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not (server and repo and run_id):
        return ""
    return f"{server}/{repo}/actions/runs/{run_id}"


def _run_checks(
    tmp_dir: str, artifact_path: str
) -> Tuple[Dict[str, object], List[str]]:
    """Run the bundle/funnel/energy stages; return ``(checks, reasons)``.

    A stage that raises :class:`VerificationStepError` stops the pipeline
    there -- a later stage needs the module the failed stage would have
    produced -- and its message becomes the sole rejection reason.
    """
    checks = dict(_EMPTY_CHECKS)
    try:
        bundle, module = load_and_build(artifact_path)
        checks["bundle"] = {
            "ok": True, "topology": bundle.manifest.get("topology")
        }
        funnel = run_funnel(bundle, module, tmp_dir)
        checks["inspect"] = funnel["inspect"]
        checks["compat"] = funnel["compat"]
        checks["drift"] = funnel["drift"]
        checks["energy"] = energy_reports(bundle.spec, module)
    except VerificationStepError as error:
        checks[error.step] = {"ok": False, "detail": error.reason}
        return checks, [f"{error.step}: {error.reason}"]
    return checks, []


def _verify(artifact_url: str) -> Tuple[Dict[str, object], List[str]]:
    """Fetch the artifact and run every check against it."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            artifact_path = fetch_artifact(
                artifact_url, os.path.join(tmp_dir, "artifact.spkf")
            )
        except VerificationStepError as error:
            return dict(_EMPTY_CHECKS), [f"{error.step}: {error.reason}"]
        return _run_checks(tmp_dir, artifact_path)


def main() -> int:
    """Verify the dispatched artifact and post its signed report."""
    version_id = _env("SPIKEFORGE_HUB_VERIFY_VERSION_ID")
    artifact_url = _env("SPIKEFORGE_HUB_VERIFY_ARTIFACT_URL")
    base_url = _env("SPIKEFORGE_HUB_API_BASE_URL")
    secret = _env("SPIKEFORGE_HUB_VERIFICATION_SECRET")

    checks, reasons = _verify(artifact_url)
    passed = not reasons
    summary = "passed every check" if passed else rejection_summary(reasons)
    report = build_report(
        version_id=version_id,
        passed=passed,
        reasons=reasons,
        checks=checks,
        summary=summary,
        run_url=_run_url(),
    )
    try:
        post_report(base_url, version_id, report, secret)
    except CallbackError as error:
        sys.exit(f"hub_verify: could not deliver the report: {error}")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
