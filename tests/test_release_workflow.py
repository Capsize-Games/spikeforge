"""Text guards on ``.github/workflows/release.yml``.

Nothing imports the release workflow, so these assertions read the YAML
as text to keep the hub wheel's runtime-data gate (issue #164) from
silently disappearing: the built wheel must be installed into a clean
venv and smoke-tested -- catalog listing plus every reference weight
resolved and checksum-verified -- before the publish action runs.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_RELEASE = _ROOT / ".github" / "workflows" / "release.yml"


def _text() -> str:
    """Return the release workflow's YAML text."""
    return _RELEASE.read_text(encoding="utf-8")


def test_release_smoke_tests_the_hub_wheel_before_publish() -> None:
    """The hub smoke gate exists and runs ahead of the PyPI publish."""
    text = _text()
    gate = text.index("Smoke-test the hub wheel")
    publish = text.index("pypa/gh-action-pypi-publish")
    assert gate < publish


def test_the_gate_installs_the_built_wheel_into_a_clean_venv() -> None:
    """The gate installs the just-built wheel, not the checkout."""
    text = _text()
    assert "python -m venv" in text
    assert "spikeforge_hub-*.whl" in text


def test_the_gate_checks_the_catalog_and_each_weight() -> None:
    """The gate lists the catalog and verifies every packaged weight."""
    text = _text()
    assert 'spikeforge-hub", "list"' in text
    assert "reference_path" in text
    assert "verify_file" in text
