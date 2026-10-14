"""Pyproject paths and pin helpers shared by the packaging tests."""

import json
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

_ROOT = Path(__file__).resolve().parent.parent
_CORE = _ROOT / "packages" / "spikeforge" / "pyproject.toml"
_SERVER = _ROOT / "packages" / "spikeforge-server" / "pyproject.toml"
_TARGETS = _ROOT / "packages" / "spikeforge-targets" / "pyproject.toml"
_HUB = _ROOT / "packages" / "spikeforge-hub" / "pyproject.toml"
_MATRIX = _ROOT / "compatibility.json"

_EXPECTED_SCRIPTS = {
    "spikeforge",
    "spikeforge-encodings",
    "spikeforge-verify",
    "spikeforge-records",
    "spikeforge-benchmark",
}
# ``web`` is intentionally absent: its dependencies are now the base
# dependencies of the ``spikeforge-server`` distribution. ``norse`` and
# ``lava`` moved to ``spikeforge-targets``; ``hub`` moved to
# ``spikeforge-hub`` — each with the code it gates.
_EXPECTED_EXTRAS = {
    "all",
    "dev",
    "nir",
    "events",
    "onnx",
    "tracking",
    "tracking-wandb",
    "docs",
}
_EXPECTED_CORE_DEPS = {
    "capsize-commons>=0.1.1",
    "torch>=2.5",
    "torchvision>=0.20",
    "snntorch>=1.0",
    "matplotlib>=3.8",
    "Pillow>=10.0",
    "numpy>=1.26",
    "psutil>=5.9",
}
# The satellite pins are deliberately absent here and asserted through
# ``_satellite_pin`` instead: a frozen literal has to be hand-edited on every
# release, and the contract that actually matters is that the pin agrees with
# the version the satellite declares, not that it equals some past number.
_EXPECTED_SERVER_DEPS = {
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "websockets>=12.0",
    "pydantic>=2.5",
}
_EXPECTED_SERVER_SCRIPTS = {
    "spikeforge-server",
}
_EXPECTED_TARGETS_SCRIPTS = {
    "spikeforge-energy",
    "spikeforge-targets",
}
_EXPECTED_TARGETS_EXTRAS = {
    "dev",
    "norse",
    "lava",
}
_EXPECTED_TARGETS_DEPS = {
    "numpy>=1.26",
    "torch>=2.5",
}
_EXPECTED_HUB_SCRIPTS = {
    # PT-W7 registry governance ships as a hub-owned script (the plan's
    # "extend hub" option) rather than a separate distribution.
    "spikeforge-hub",
    "spikeforge-registry",
}
_EXPECTED_HUB_EXTRAS = {
    "dev",
}
_EXPECTED_HUB_DEPS = {
    "torch>=2.5",
    "huggingface_hub>=0.20",
}
_CLIENTS = _ROOT / "packages" / "spikeforge-clients" / "pyproject.toml"
_EXPECTED_CLIENTS_SCRIPTS = {
    "spikeforge-clients",
}
_IO = _ROOT / "packages" / "spikeforge-io" / "pyproject.toml"
_EXPECTED_IO_SCRIPTS = {
    "spikeforge-io",
}
_EXPECTED_IO_DEPS = {
    "torch>=2.5",
    "numpy>=1.26",
}


def _pyproject(path: Path) -> Dict[str, Any]:
    """Parse a PEP 621 ``pyproject.toml`` with the stdlib ``tomllib``."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _core_pin() -> str:
    """Return the current compatible-release pin for satellite tests."""
    version = _pyproject(_CORE)["project"]["version"]
    return f"spikeforge~={version}"


def _satellite_pin(pyproject: Path) -> str:
    """Return the compatible-release pin a dependent must declare.

    Derived from the distribution's own declared version so the pin and the
    version cannot drift apart across a release; ``compatibility.json`` and
    ``scripts/check_packaging_guards.py`` police the matrix itself.
    """
    project = _pyproject(pyproject)["project"]
    return f"{project['name']}~={project['version']}"


def _matrix_version(distribution: str) -> str:
    """Return ``distribution``'s version in the newest compatibility row."""
    with _MATRIX.open("rb") as handle:
        releases = json.load(handle)["releases"]
    return str(releases[-1][distribution])
