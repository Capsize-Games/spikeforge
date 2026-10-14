"""Text guards on ``.github/workflows/hub-verify.yml``.

Nothing imports the verification workflow, so these assertions read the
YAML as text to keep the wiring preflight (issue #162) from silently
disappearing: the job must refuse to build the sandbox when the
callback base URL, the signing secret, or the dispatch payload is
missing, failing before the torch install rather than after it.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_VERIFY = _ROOT / ".github" / "workflows" / "hub-verify.yml"


def _text() -> str:
    """Return the verification workflow's YAML text."""
    return _VERIFY.read_text(encoding="utf-8")


def test_verify_confirms_its_wiring_before_installing_torch() -> None:
    """The preflight exists and runs ahead of the torch install."""
    text = _text()
    preflight = text.index("Confirm the runner is wired")
    install = text.index("Install torch CPU wheels")
    assert preflight < install


def test_the_preflight_covers_every_required_variable() -> None:
    """The preflight checks the secret, the variable, and the payload."""
    text = _text()
    for name in (
        "SPIKEFORGE_HUB_API_BASE_URL",
        "SPIKEFORGE_HUB_VERIFICATION_SECRET",
        "SPIKEFORGE_HUB_VERIFY_VERSION_ID",
        "SPIKEFORGE_HUB_VERIFY_ARTIFACT_URL",
    ):
        assert name in text
