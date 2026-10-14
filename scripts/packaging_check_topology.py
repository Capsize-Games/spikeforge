"""Import-root and console-script ownership checks.

Split out of ``check_packaging_guards.py`` (#149).
"""

from pathlib import Path
from typing import Dict, List, Optional, Set

try:  # imported as part of the scripts package
    from scripts.packaging_layout import _scripts
except ImportError:  # run directly as scripts/*.py
    from packaging_layout import _scripts


def check_disjoint_roots(
    roots_by_distribution: Dict[str, Set[str]],
) -> Optional[str]:
    """Return an error when two distributions ship the same import root."""
    names = sorted(roots_by_distribution)
    clashes: List[str] = []
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = sorted(
                roots_by_distribution[left] & roots_by_distribution[right]
            )
            if overlap:
                clashes.append(
                    f"{', '.join(overlap)} shared by {left} and {right}"
                )
    if clashes:
        return "import roots shipped by multiple distributions: " + (
            "; ".join(clashes)
        )
    return None


def check_script_ownership(
    pyprojects: Dict[str, Path],
) -> Optional[str]:
    """Return an error when a console script is owned by two distributions."""
    owners: Dict[str, List[str]] = {}
    for distribution, path in pyprojects.items():
        for script in _scripts(path):
            owners.setdefault(script, []).append(distribution)
    clashes = {name: names for name, names in owners.items() if len(names) > 1}
    if not clashes:
        return None
    detail = "; ".join(
        f"{name} in {', '.join(sorted(names))}"
        for name, names in sorted(clashes.items())
    )
    return f"console scripts owned by multiple distributions: {detail}"
