"""The release core-pin rule shared by CI and the local packaging guards.

The ``release.yml`` "Verify the tag against pyproject and compatibility.json"
step and ``scripts/check_packaging_guards.py`` implement the same rule. The
workflow shell is embedded in YAML and cannot be imported directly, so these
tests lock the shared helper (and assert the workflow keeps the mirrored
allowlist) instead.
"""

import json
from pathlib import Path

from scripts.check_packaging_guards import (
    CLIENTS_DISTRIBUTION,
    CORE_DISTRIBUTION,
    CORE_FREE_DISTRIBUTIONS,
    check_core_pin,
)

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

_ROOT = Path(__file__).resolve().parent.parent
_RELEASE = _ROOT / ".github" / "workflows" / "release.yml"


def test_core_free_distributions_contains_clients() -> None:
    """The client distribution is the declared core-free exemption."""
    assert CLIENTS_DISTRIBUTION in CORE_FREE_DISTRIBUTIONS


def test_core_free_distribution_is_exempt_from_the_core_pin() -> None:
    """``spikeforge-clients`` declares no core dependency and is accepted."""
    assert check_core_pin(CLIENTS_DISTRIBUTION, [], "0.3.0") is None


def test_core_free_distribution_with_a_core_dependency_is_rejected() -> None:
    """A core-free distribution may not smuggle in a core pin."""
    problem = check_core_pin(
        CLIENTS_DISTRIBUTION, ["spikeforge~=0.3.0"], "0.3.0"
    )
    assert problem is not None


def test_non_core_distribution_with_a_missing_core_pin_is_rejected() -> None:
    """A hypothetical satellite without a pin is refused."""
    problem = check_core_pin("spikeforge-hypothetical", [], "0.3.0")
    assert problem is not None


def test_non_core_distribution_with_a_wrong_core_pin_is_rejected() -> None:
    """A hypothetical satellite on the wrong core line is refused."""
    problem = check_core_pin(
        "spikeforge-hypothetical", ["spikeforge~=0.2.0"], "0.3.0"
    )
    assert problem is not None


def test_non_core_distribution_with_the_matrix_pin_is_accepted() -> None:
    """A satellite pinned to the matrix core version is accepted."""
    assert (
        check_core_pin(
            "spikeforge-hypothetical",
            ["numpy>=1.26", "spikeforge~=0.3.0"],
            "0.3.0",
        )
        is None
    )


def test_core_distribution_itself_is_exempt() -> None:
    """Core carries no self-pin and is always accepted."""
    assert check_core_pin(CORE_DISTRIBUTION, ["torch>=2.5"], "0.3.0") is None


def test_release_workflow_mirrors_the_core_free_allowlist() -> None:
    """release.yml documents the same core-free exemption as the guard."""
    text = _RELEASE.read_text(encoding="utf-8")
    assert "core_free_distributions" in text
    assert "core-free" in text
    assert CLIENTS_DISTRIBUTION in text


def test_shipped_pyprojects_satisfy_the_core_pin_rule() -> None:
    """Every real distribution agrees with the shared rule and the matrix."""
    matrix = json.loads(
        (_ROOT / "compatibility.json").read_text(encoding="utf-8")
    )
    core_version = matrix["releases"][-1][CORE_DISTRIBUTION]

    checked = 0
    for pyproject in sorted((_ROOT / "packages").glob("*/pyproject.toml")):
        with pyproject.open("rb") as handle:
            project = tomllib.load(handle)["project"]
        checked += 1
        problem = check_core_pin(
            project["name"],
            project.get("dependencies", []),
            core_version,
        )
        assert problem is None, f"{pyproject}: {problem}"
    assert checked >= 7
