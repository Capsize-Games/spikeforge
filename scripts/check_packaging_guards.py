#!/usr/bin/env python3
"""Packaging-topology guards for the ARCH-0001 multi-distribution layout.

Three release-blocking health metrics from
``plans/arch-0001-decision-metrics.md`` that the runtime blockers do not cover:

* **disjoint import roots** - no top-level import root may be shipped by two
  distributions (the PEP 420 / packaging-profile health metric);
* **console-script ownership** - no console-script name may appear in two
  distributions;
* **compatibility pin** - every satellite's ``snn-interpreter~=X.Y.0`` pin must
  equal the core version recorded in ``compatibility.json``.

Import roots and scripts are read from each distribution's ``pyproject.toml``.
When built wheels are passed with ``--core-wheel`` and ``--server-wheel`` the
import-root check runs against the actual archives instead, so CI proves the
artifact and not just the intent.

Usage::

    python scripts/check_packaging_guards.py \
        --core-wheel dist/snn_interpreter-*.whl \
        --server-wheel dist/snn_interpreter_server-*.whl
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

#: Repository root (this file lives in ``scripts/``).
REPO_ROOT = Path(__file__).resolve().parent.parent

CORE_DISTRIBUTION = "snn-interpreter"
SERVER_DISTRIBUTION = "snn-interpreter-server"
CORE_PYPROJECT = REPO_ROOT / "packages" / CORE_DISTRIBUTION / "pyproject.toml"
SERVER_PYPROJECT = (
    REPO_ROOT / "packages" / SERVER_DISTRIBUTION / "pyproject.toml"
)
COMPATIBILITY = REPO_ROOT / "compatibility.json"


def _read_toml(path: Path) -> Dict[str, object]:
    """Parse a ``pyproject.toml`` into a dict."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _project(path: Path) -> Dict[str, object]:
    """Return the ``[project]`` table of a ``pyproject.toml``."""
    return _read_toml(path).get("project", {})  # type: ignore[return-value]


def _scripts(path: Path) -> Set[str]:
    """Return the console-script names a distribution declares."""
    return set(_project(path).get("scripts", {}))


def _find_table(path: Path) -> Dict[str, object]:
    """Return the ``setuptools.packages.find`` table, or an empty dict."""
    setuptools = _read_toml(path).get("tool", {}).get("setuptools", {})
    if not isinstance(setuptools, dict):
        return {}
    find = setuptools.get("packages", {}).get("find", {})
    return find if isinstance(find, dict) else {}


def _py_modules(path: Path) -> List[str]:
    """Return the ``setuptools.py-modules`` list, or an empty list."""
    setuptools = _read_toml(path).get("tool", {}).get("setuptools", {})
    if not isinstance(setuptools, dict):
        return []
    modules = setuptools.get("py-modules", [])
    return list(modules) if isinstance(modules, list) else []


def roots_from_pyproject(path: Path) -> Set[str]:
    """Return the top-level import roots a distribution declares."""
    roots = {module.split(".")[0] for module in _py_modules(path)}
    for pattern in _find_table(path).get("include", []):
        roots.add(str(pattern).rstrip("*").split(".")[0])
    return roots


def roots_from_wheel(path: Path) -> Set[str]:
    """Return the top-level archive entries of a built wheel."""
    roots: Set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            top = name.split("/", 1)[0]
            if top.endswith((".dist-info", ".data")):
                continue
            roots.add(top)
    return roots


def check_disjoint_roots(
    core_roots: Set[str], server_roots: Set[str]
) -> Optional[str]:
    """Return an error message when the import roots overlap, else None."""
    overlap = sorted(core_roots & server_roots)
    if overlap:
        return f"import roots shared by core and server: {', '.join(overlap)}"
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


def _latest_release() -> Dict[str, object]:
    """Return the newest release entry from ``compatibility.json``."""
    with COMPATIBILITY.open("rb") as handle:
        data = json.load(handle)
    releases = data.get("releases", [])
    if not releases:
        raise SystemExit("compatibility.json has no release entries")
    return releases[-1]


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
        expected_pin = (
            f"{CORE_DISTRIBUTION}~={release.get(CORE_DISTRIBUTION)}"
        )
        for dependency in project.get("dependencies", []):
            if (
                str(dependency).startswith(CORE_DISTRIBUTION)
                and dependency != expected_pin
            ):
                problems.append(
                    f"{distribution} pin {dependency!r} != "
                    f"{expected_pin!r}"
                )
    if problems:
        return "compatibility pin mismatch: " + "; ".join(problems)
    return None


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the ARCH-0001 packaging-topology guards."
    )
    parser.add_argument("--core-wheel", type=Path, default=None)
    parser.add_argument("--server-wheel", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the guards and return a process exit code."""
    args = _parse_args(argv)
    pyprojects = {
        CORE_DISTRIBUTION: CORE_PYPROJECT,
        SERVER_DISTRIBUTION: SERVER_PYPROJECT,
    }
    if args.core_wheel and args.server_wheel:
        core_roots = roots_from_wheel(args.core_wheel)
        server_roots = roots_from_wheel(args.server_wheel)
        source = "built wheels"
    else:
        core_roots = roots_from_pyproject(CORE_PYPROJECT)
        server_roots = roots_from_pyproject(SERVER_PYPROJECT)
        source = "pyproject discovery"
    checks = (
        (
            "disjoint import roots",
            check_disjoint_roots(core_roots, server_roots),
        ),
        ("console-script ownership", check_script_ownership(pyprojects)),
        ("compatibility pin", check_matrix_pins(pyprojects)),
    )
    failures = [(name, message) for name, message in checks if message]
    if failures:
        for name, message in failures:
            print(f"FAIL {name}: {message}", file=sys.stderr)
        return 1
    print(
        "packaging-guards: OK - "
        + ", ".join(name for name, _ in checks)
        + f" ({source})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
