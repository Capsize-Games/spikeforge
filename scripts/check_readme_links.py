#!/usr/bin/env python3
"""Fail when the packaged README would render broken links on PyPI.

Every distribution in ``packages/`` sets ``readme = "README.md"``, and in each
case that file is a committed symlink to the repository-root README. GitHub
resolves a relative link against the repository; **PyPI does not**. A relative
``href`` or ``src`` that reads perfectly on GitHub is therefore a dead link --
or, for the hero image, a broken-image icon -- on all seven project pages.

This gate renders the README through ``readme_renderer``, the exact library
PyPI uses to turn ``long_description`` into the project page, and fails on any
``href``/``src`` that is not absolute. It additionally resolves every
``blob/main`` and ``raw`` URL back to a path in this repository, so a typo in
an absolute link is caught here rather than by a visitor, and it asserts that
each distribution really does package the root README.

Usage::

    pip install "readme_renderer[md]"
    python scripts/check_readme_links.py

Intended to run in the ``published-surfaces`` CI job, which needs no torch.
"""

import re
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
PACKAGES = REPO_ROOT / "packages"

#: The repository URL prefixes that the README is allowed to point at, mapped
#: to the in-repository path each one addresses. Anything matching these is
#: resolved on disk so a broken absolute link fails here too.
REPO_URL_PREFIXES = (
    "https://github.com/Capsize-Games/spikeforge/blob/main/",
    "https://github.com/Capsize-Games/spikeforge/tree/main/",
    "https://raw.githubusercontent.com/Capsize-Games/spikeforge/main/",
)

#: Attribute values that are fine to leave relative: pure fragments are
#: in-page anchors, which PyPI resolves correctly.
_ABSOLUTE = ("http://", "https://", "mailto:", "#")

_ATTRIBUTE = re.compile(r'\b(href|src)="([^"]*)"')


def _render() -> str:
    """Render the README exactly the way PyPI renders long_description."""
    try:
        from readme_renderer.markdown import render
    except ImportError:  # pragma: no cover - CI installs the dependency
        print(
            "readme_renderer is not installed: "
            'pip install "readme_renderer[md]"',
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    html = render(README.read_text(encoding="utf-8"))
    if html is None:
        print(
            "readme_renderer refused the README: PyPI would show the raw "
            "text instead of a rendered page.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return html


def _relative_references(html: str) -> List[str]:
    """Return every rendered attribute PyPI would fail to resolve."""
    return [
        f"{attribute}={value}"
        for attribute, value in _ATTRIBUTE.findall(html)
        if not value.startswith(_ABSOLUTE)
    ]


def _unresolvable_repository_links(html: str) -> List[str]:
    """Return every repository URL that names a path which does not exist."""
    missing: List[str] = []
    for attribute, value in _ATTRIBUTE.findall(html):
        for prefix in REPO_URL_PREFIXES:
            if not value.startswith(prefix):
                continue
            path = value[len(prefix):].split("#", 1)[0].split("?", 1)[0]
            if path and not (REPO_ROOT / path).exists():
                missing.append(f"{attribute}={value}")
            break
    return missing


def _unpackaged_distributions() -> List[str]:
    """Return distributions whose README is not the repository-root one."""
    unpackaged: List[str] = []
    for pyproject in sorted(PACKAGES.glob("*/pyproject.toml")):
        packaged = pyproject.parent / "README.md"
        if not packaged.exists():
            unpackaged.append(f"{pyproject.parent.name}: no README.md")
        elif packaged.resolve() != README.resolve():
            unpackaged.append(
                f"{pyproject.parent.name}: README.md is not the root README"
            )
    return unpackaged


def main() -> int:
    """Render the README and report anything PyPI would show as broken."""
    html = _render()
    failures = 0

    relative = _relative_references(html)
    for reference in relative:
        print(f"relative reference on the PyPI page: {reference}")
    failures += len(relative)

    missing = _unresolvable_repository_links(html)
    for reference in missing:
        print(f"repository link points at a missing path: {reference}")
    failures += len(missing)

    unpackaged = _unpackaged_distributions()
    for detail in unpackaged:
        print(f"distribution does not package the root README: {detail}")
    failures += len(unpackaged)

    if failures:
        print(f"\n{failures} problem(s) would be visible on PyPI.")
        return 1
    distributions = len(list(PACKAGES.glob("*/pyproject.toml")))
    print(
        f"PyPI README OK: no relative references, every repository link "
        f"resolves, {distributions} distributions package it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
