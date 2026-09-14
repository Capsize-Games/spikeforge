#!/usr/bin/env python3
"""Smoke-test every declared console script against a minimal install.

A console script that is broken on a bare ``pip install spikeforge`` -- no
``[all]``, no ``[dev]`` -- is invisible to every other CI job, because they
all install the extras together. This script closes that gap: it reads the
``[project.scripts]`` table straight from
``packages/spikeforge/pyproject.toml`` (so a newly added script is covered
automatically) and runs each one with ``--help``, asserting a clean, prompt
exit 0 and no output on stderr.

Usage::

    python scripts/check_console_scripts.py

Intended to run in a virtualenv that has installed *only* the core
``spikeforge`` distribution with no extras (see the ``headless`` CI job).
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, NamedTuple

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

#: Repository root (this file lives in ``scripts/``).
REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_PYPROJECT = REPO_ROOT / "packages" / "spikeforge" / "pyproject.toml"
#: Generous, but bounded: a broken script must not hang CI, and none of
#: these commands should ever legitimately touch the network or train.
TIMEOUT_SECONDS = 30


class Result(NamedTuple):
    """The outcome of running one console script with ``--help``."""

    name: str
    ok: bool
    detail: str


def _declared_scripts() -> List[str]:
    """Return the console-script names from the core distribution."""
    with CORE_PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    scripts = data.get("project", {}).get("scripts", {})
    return sorted(scripts)


def _check(name: str) -> Result:
    """Run ``name --help`` and report whether it exited cleanly."""
    path = shutil.which(name)
    if path is None:
        return Result(name, False, "not found on PATH after install")
    try:
        proc = subprocess.run(
            [path, "--help"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return Result(
            name,
            False,
            f"timed out after {TIMEOUT_SECONDS}s (hung, or reached the "
            "network instead of printing help)",
        )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        detail = tail[-1] if tail else "no output"
        return Result(name, False, f"exit {proc.returncode}: {detail}")
    return Result(name, True, "ok")


def main() -> int:
    """Run every declared console script and report pass/fail."""
    names = _declared_scripts()
    if not names:
        print(
            "check_console_scripts: no [project.scripts] found",
            file=sys.stderr,
        )
        return 1
    results = [_check(name) for name in names]
    failures = [result for result in results if not result.ok]
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"{status:4s} {result.name}: {result.detail}")
    if failures:
        print(
            f"\ncheck_console_scripts: {len(failures)}/{len(results)} "
            "console script(s) broken on a minimal install",
            file=sys.stderr,
        )
        return 1
    print(f"\ncheck_console_scripts: OK - {len(results)} console script(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
