"""Repository layout constants and TOML discovery for the guards.

Single home for the distribution names, their ``pyproject.toml`` paths,
and the ``[project]``/setuptools readers the topology and pin checks
share. Split out of ``check_packaging_guards.py`` (#149).
"""

import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Set

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

#: Repository root (this file lives in ``scripts/``).
REPO_ROOT = Path(__file__).resolve().parent.parent

CORE_DISTRIBUTION = "spikeforge"
SERVER_DISTRIBUTION = "spikeforge-server"
SERVE_DISTRIBUTION = "spikeforge-serve"
CLIENTS_DISTRIBUTION = "spikeforge-clients"
TARGETS_DISTRIBUTION = "spikeforge-targets"
HUB_DISTRIBUTION = "spikeforge-hub"
IO_DISTRIBUTION = "spikeforge-io"
CORE_PYPROJECT = REPO_ROOT / "packages" / CORE_DISTRIBUTION / "pyproject.toml"
SERVER_PYPROJECT = (
    REPO_ROOT / "packages" / SERVER_DISTRIBUTION / "pyproject.toml"
)
SERVE_PYPROJECT = (
    REPO_ROOT / "packages" / SERVE_DISTRIBUTION / "pyproject.toml"
)
CLIENTS_PYPROJECT = (
    REPO_ROOT / "packages" / CLIENTS_DISTRIBUTION / "pyproject.toml"
)
TARGETS_PYPROJECT = (
    REPO_ROOT / "packages" / TARGETS_DISTRIBUTION / "pyproject.toml"
)
HUB_PYPROJECT = REPO_ROOT / "packages" / HUB_DISTRIBUTION / "pyproject.toml"
IO_PYPROJECT = REPO_ROOT / "packages" / IO_DISTRIBUTION / "pyproject.toml"
COMPATIBILITY = REPO_ROOT / "compatibility.json"

#: Distributions that deliberately declare no ``spikeforge`` core dependency.
#: A client install must never pull the torch-backed core, so these are exempt
#: from the core-pin rule; declaring a core dependency in one is itself an
#: error. Keep this in sync with ``core_free_distributions`` in
#: ``.github/workflows/release.yml``.
CORE_FREE_DISTRIBUTIONS = frozenset({CLIENTS_DISTRIBUTION})

#: Satellites that must ride along a core MINOR bump with their own
#: release: every non-core distribution except the core-free ones.
PINNED_DISTRIBUTIONS = (
    SERVER_DISTRIBUTION,
    SERVE_DISTRIBUTION,
    TARGETS_DISTRIBUTION,
    HUB_DISTRIBUTION,
    IO_DISTRIBUTION,
)


def _read_toml(path: Path) -> Dict[str, object]:
    """Parse a ``pyproject.toml`` into a dict."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _project(path: Path) -> Dict[str, object]:
    """Return the ``[project]`` table of a ``pyproject.toml``."""
    project = _read_toml(path).get("project", {})
    if not isinstance(project, dict):
        return {}
    return project


def _scripts(path: Path) -> Set[str]:
    """Return the console-script names a distribution declares."""
    scripts = _project(path).get("scripts", {})
    if not isinstance(scripts, dict):
        return set()
    return set(scripts)


def _dependencies(project: Dict[str, object]) -> List[object]:
    """Return the ``dependencies`` list of a ``[project]`` table."""
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list):
        return []
    return dependencies


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
