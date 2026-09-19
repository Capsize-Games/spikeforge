"""The verification report's shape and the hub-api trust contract."""

from hub_verify.report import build_report, rejection_summary


def test_passed_report_carries_a_true_boolean() -> None:
    """``passed`` is the JSON boolean hub-api's ``trust.label_for`` reads.

    ``hub_api/catalog/trust.py::label_for`` only shows ``machine-checked``
    when ``version.verification["passed"] is True`` -- not truthy, the
    literal boolean -- so this is the one field this report must never get
    wrong.
    """
    report = build_report(
        version_id="v1",
        passed=True,
        reasons=[],
        checks={"bundle": {"ok": True}},
        summary="passed every check",
        run_url="https://github.com/x/y/actions/runs/1",
    )
    assert report["passed"] is True
    assert report["reasons"] == []
    assert report["version_id"] == "v1"


def test_failed_report_names_the_reason() -> None:
    """A rejection carries the actionable reason, not just a flag."""
    reasons = ["bundle: weights are not loadable: bad magic number"]
    report = build_report(
        version_id="v2",
        passed=False,
        reasons=reasons,
        checks={"bundle": {"ok": False}},
        summary=rejection_summary(reasons),
        run_url="",
    )
    assert report["passed"] is False
    assert "weights are not loadable" in report["summary"]


def test_rejection_summary_of_no_reasons_names_the_bug() -> None:
    """An empty reason list on a rejection is itself flagged, not hidden."""
    assert "bug" in rejection_summary([])
