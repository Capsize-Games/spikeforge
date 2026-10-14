#!/usr/bin/env python3
"""Static core-boundary scan for the ``spikeforge`` distribution.

ARCH-0001 Phase 1c. The core distribution must install and import without any
of the forbidden root packages (see ``plans/arch-0001-core-boundary.md``). This
script enforces the *static* half of that decision: it parses every module the
core distribution ships and fails on

* a **module-level** ``import``/``from`` of a forbidden root, anywhere, and
* a **function-local** import of a forbidden root outside the enumerated
  lazy-shim allow-list.

The runtime half is the extended ``blocked-deps`` gate; the artifact half is
the ``headless`` CI job. This script is the cheap, fast, offline half.

Usage::

    python scripts/check_core_boundary.py [--quiet]

It exits 0 when the tree is clean and 1 (with a per-violation report)
otherwise. Each report line is ``path:line: kind import of forbidden root
'root'`` so a reviewer can jump straight to the offending statement.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from check_core_boundary_rules import (
    ALLOWED_LOCAL_IMPORTERS,
    FORBIDDEN_ROOTS,
    FROZEN_BOUNDARY,
    Violation,
)
from check_core_boundary_scan import (
    format_violation,
    iter_modules,
    scan,
)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fail when a core-distribution module imports a forbidden root."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print nothing on success (violations are always reported)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the frozen boundary and the lazy-shim allow-list",
    )
    return parser.parse_args(argv)


def _print_boundary() -> None:
    """Print the frozen boundary and the lazy-shim allow-list."""
    print("forbidden boundary (design): " + ", ".join(FROZEN_BOUNDARY))
    print("forbidden import roots (scanned): " + ", ".join(
        sorted(FORBIDDEN_ROOTS)
    ))
    print("lazy-shim allow-list: " + ", ".join(
        sorted(ALLOWED_LOCAL_IMPORTERS)
    ))


def _report_violations(
    violations: List[Violation], modules: List[Path]
) -> int:
    """Print every violation and return the failure exit code."""
    for violation in violations:
        print(format_violation(violation), file=sys.stderr)
    print(
        f"core-boundary: {len(violations)} violation(s) across "
        f"{len(modules)} module(s)",
        file=sys.stderr,
    )
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the scan and return a process exit code."""
    args = _parse_args(argv)
    if args.list:
        _print_boundary()
        return 0
    modules = iter_modules()
    violations = scan(modules)
    if violations:
        return _report_violations(violations, modules)
    if not args.quiet:
        print(
            f"core-boundary: OK - {len(modules)} module(s) scanned, "
            "no forbidden imports"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
