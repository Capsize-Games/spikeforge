"""CLI orchestration: env reading, stage sequencing, always-posts-a-report."""

from typing import Any, Dict, List, Tuple

import pytest
from hub_verify import cli
from hub_verify.errors import CallbackError, VerificationStepError

_ENV = {
    "SPIKEFORGE_HUB_VERIFY_VERSION_ID": "v1",
    "SPIKEFORGE_HUB_VERIFY_ARTIFACT_URL": "https://cdn.example/a.spkf",
    "SPIKEFORGE_HUB_API_BASE_URL": "https://hub.example",
    "SPIKEFORGE_HUB_VERIFICATION_SECRET": "sekrit",
}


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate every required environment variable for ``main()``."""
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)


class _FakeBundle:
    """A stand-in with just the attributes ``_run_checks`` reads."""

    manifest = {"topology": "fc_small"}
    spec = None


def test_missing_env_var_exits_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing required setting exits rather than crashing obscurely."""
    monkeypatch.delenv(
        "SPIKEFORGE_HUB_VERIFY_VERSION_ID", raising=False
    )
    with pytest.raises(SystemExit, match="SPIKEFORGE_HUB_VERIFY_VERSION_ID"):
        cli.main()


def test_a_passing_artifact_posts_passed_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every stage succeeding posts ``passed: true`` with no reasons."""
    _set_env(monkeypatch)
    monkeypatch.setattr(
        cli, "fetch_artifact", lambda url, dest: dest
    )
    monkeypatch.setattr(
        cli, "load_and_build", lambda path: (_FakeBundle(), object())
    )
    monkeypatch.setattr(
        cli, "run_funnel",
        lambda bundle, module, tmp: {
            "inspect": {"kind": "state_dict"},
            "compat": {"verdict": "incompatible"},
            "drift": {"within_tolerance": True},
        },
    )
    monkeypatch.setattr(cli, "energy_reports", lambda spec, module: [])

    posted: Dict[str, Any] = {}
    monkeypatch.setattr(
        cli,
        "post_report",
        lambda base, vid, report, secret: posted.update(report=report),
    )
    assert cli.main() == 0
    assert posted["report"]["passed"] is True
    assert posted["report"]["reasons"] == []


def test_a_failing_stage_posts_the_named_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage failure posts ``passed: false`` naming which step and why."""
    _set_env(monkeypatch)
    monkeypatch.setattr(cli, "fetch_artifact", lambda url, dest: dest)

    def _raise(path: str) -> Tuple[Any, Any]:
        raise VerificationStepError("bundle", "weights are not loadable")

    monkeypatch.setattr(cli, "load_and_build", _raise)

    posted: Dict[str, Any] = {}
    monkeypatch.setattr(
        cli,
        "post_report",
        lambda base, vid, report, secret: posted.update(report=report),
    )
    assert cli.main() == 0
    assert posted["report"]["passed"] is False
    reasons: List[str] = posted["report"]["reasons"]
    assert reasons == ["bundle: weights are not loadable"]


def test_a_failed_callback_exits_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The job fails when the signed report cannot be delivered at all."""
    _set_env(monkeypatch)
    monkeypatch.setattr(cli, "fetch_artifact", lambda url, dest: dest)
    monkeypatch.setattr(
        cli, "load_and_build", lambda path: (_FakeBundle(), object())
    )
    monkeypatch.setattr(
        cli, "run_funnel",
        lambda bundle, module, tmp: {
            "inspect": {}, "compat": {}, "drift": {"within_tolerance": True}
        },
    )
    monkeypatch.setattr(cli, "energy_reports", lambda spec, module: [])

    def _raise(*_args: Any) -> None:
        raise CallbackError("hub API unreachable")

    monkeypatch.setattr(cli, "post_report", _raise)
    with pytest.raises(SystemExit, match="could not deliver"):
        cli.main()
