"""Core-pin checks for core-requiring and core-free distributions.

The rule mirrors the ``release.yml`` verify step so local guards and CI
agree. Split out of ``check_packaging_guards.py`` (#149).
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:  # imported as part of the scripts package
    from scripts.packaging_check_bump import _latest_release
    from scripts.packaging_layout import (
        CORE_DISTRIBUTION,
        CORE_FREE_DISTRIBUTIONS,
        _dependencies,
        _project,
    )
except ImportError:  # run directly as scripts/*.py
    from packaging_check_bump import _latest_release
    from packaging_layout import (
        CORE_DISTRIBUTION,
        CORE_FREE_DISTRIBUTIONS,
        _dependencies,
        _project,
    )


def _requirement_name(requirement: object) -> str:
    """Return the distribution name of a PEP 508 requirement string.

    The satellites share the ``spikeforge`` prefix (for example
    ``spikeforge-targets``), so a plain ``str.startswith`` would misread a
    satellite pin as a core pin; comparing the bare name avoids that.
    """
    text = str(requirement).strip()
    for index, char in enumerate(text):
        if char in "<>=!~;[":
            return text[:index].strip()
    return text


def check_core_pin(
    distribution: str,
    dependencies: Sequence[object],
    core_version: object,
) -> Optional[str]:
    """Return an error when a distribution's core pin breaks the rule.

    The rule mirrors the ``release.yml`` verify step so local guards and CI
    agree:

    * the core distribution itself is exempt;
    * a distribution in :data:`CORE_FREE_DISTRIBUTIONS` must declare no
      ``spikeforge`` dependency at all;
    * every other non-core distribution must declare exactly one
      ``spikeforge`` dependency pinned ``spikeforge~=<core_version>``.
    """
    if distribution == CORE_DISTRIBUTION:
        return None
    expected_pin = f"{CORE_DISTRIBUTION}~={core_version}"
    pins = [
        dependency
        for dependency in dependencies
        if _requirement_name(dependency) == CORE_DISTRIBUTION
    ]
    if distribution in CORE_FREE_DISTRIBUTIONS:
        if pins:
            return (
                f"{distribution} is declared core-free but declares core "
                f"dependencies {pins}"
            )
        return None
    if pins != [expected_pin]:
        return (
            f"{distribution} core pin {pins} is not the matrix pin "
            f"{expected_pin!r}"
        )
    return None


def check_matrix_pins(pyprojects: Dict[str, Path]) -> Optional[str]:
    """Return an error when a distribution disagrees with the matrix."""
    release = _latest_release()
    problems: List[str] = []
    for distribution, path in pyprojects.items():
        project = _project(path)
        expected_version = release.get(distribution)
        actual_version = project.get("version")
        if expected_version != actual_version:
            problems.append(
                f"{distribution} version {actual_version!r} != "
                f"matrix {expected_version!r}"
            )
        if distribution == CORE_DISTRIBUTION:
            continue
        problem = check_core_pin(
            distribution,
            _dependencies(project),
            release.get(CORE_DISTRIBUTION),
        )
        if problem:
            problems.append(problem)
    if problems:
        return "compatibility pin mismatch: " + "; ".join(problems)
    return None
