"""Enumerate the shipped modules and scan them for violations."""

import ast
from pathlib import Path
from typing import List, Sequence

from check_core_boundary_rules import (
    EXCLUDED_PREFIXES,
    REPO_ROOT,
    SCAN_DIRS,
    SCAN_FILES,
    Violation,
)
from check_core_boundary_visit import _Visitor


def _module_name(path: Path) -> str:
    """Return the dotted module name for a scanned file."""
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_excluded(relative: Path) -> bool:
    """Return True when a repo-relative path is outside the core ship set."""
    return any(
        part.startswith(EXCLUDED_PREFIXES) for part in relative.parts
    )


def iter_modules() -> List[Path]:
    """Return the ``.py`` files the core distribution ships, sorted."""
    modules: List[Path] = []
    for directory in SCAN_DIRS:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if _is_excluded(path.relative_to(REPO_ROOT)):
                continue
            modules.append(path)
    for name in SCAN_FILES:
        path = REPO_ROOT / name
        if path.is_file():
            modules.append(path)
    return modules


def scan(paths: Sequence[Path]) -> List[Violation]:
    """Parse ``paths`` and return every core-boundary violation found."""
    violations: List[Violation] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        _Visitor(_module_name(path), path, violations).visit(tree)
    return violations


def format_violation(violation: Violation) -> str:
    """Render one violation as a ``path:line`` report line."""
    relative = violation.path.relative_to(REPO_ROOT)
    return (
        f"{relative}:{violation.line}: {violation.kind} import of "
        f"forbidden root '{violation.root}' ({violation.statement})"
    )
