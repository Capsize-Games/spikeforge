"""Packaging extras/console scripts and docker-compose profile config."""

from pathlib import Path
from typing import Any, Dict, Set

import pytest

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

_ROOT = Path(__file__).resolve().parent.parent
_CORE = _ROOT / "packages" / "snn-interpreter" / "pyproject.toml"
_SERVER = _ROOT / "packages" / "snn-interpreter-server" / "pyproject.toml"

_EXPECTED_SCRIPTS = {
    "snn-interpreter",
    "snn-interpreter-encodings",
    "snn-verify",
    "snn-records",
    "snn-targets",
    "snn-hub",
    "snn-energy",
    "snn-benchmark",
}
# ``web`` is intentionally absent: its dependencies are now the base
# dependencies of the ``snn-interpreter-server`` distribution.
_EXPECTED_EXTRAS = {
    "dev",
    "nir",
    "events",
    "onnx",
    "hub",
    "tracking",
    "tracking-wandb",
    "docs",
    "norse",
    "lava",
}
_EXPECTED_CORE_DEPS = {
    "torch>=2.5",
    "torchvision>=0.20",
    "snntorch>=1.0",
    "matplotlib>=3.8",
    "Pillow>=10.0",
    "numpy>=1.26",
    "psutil>=5.9",
}
_EXPECTED_SERVER_DEPS = {
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "websockets>=12.0",
    "pydantic>=2.5",
}


def _pyproject(path: Path) -> Dict[str, Any]:
    """Parse a PEP 621 ``pyproject.toml`` with the stdlib ``tomllib``."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_core_declares_expected_extras_and_scripts() -> None:
    """Every advertised core extra and console script is declared."""
    project = _pyproject(_CORE)["project"]
    assert set(project["optional-dependencies"]) == _EXPECTED_EXTRAS
    assert set(project["scripts"]) == _EXPECTED_SCRIPTS


def test_core_no_longer_ships_the_web_extra() -> None:
    """The server stack is owned by the server distribution, not core."""
    extras = _pyproject(_CORE)["project"]["optional-dependencies"]
    assert "web" not in extras


def test_core_dependencies_match_the_design() -> None:
    """Core declares exactly the headless-safe dependency set."""
    deps = _pyproject(_CORE)["project"]["dependencies"]
    assert set(deps) == _EXPECTED_CORE_DEPS


def test_console_script_targets_are_stable() -> None:
    """The eight entry points resolve to their unchanged targets."""
    scripts = _pyproject(_CORE)["project"]["scripts"]
    assert scripts["snn-interpreter"] == "main:main"
    assert scripts["snn-interpreter-encodings"] == "main_encodings:main"
    assert scripts["snn-verify"] == "snn_interpreter.cli.verify:main"
    assert scripts["snn-records"] == "snn_interpreter.cli.records_cli:main"
    assert scripts["snn-targets"] == "snn_interpreter.cli.target_cli:main"
    assert scripts["snn-hub"] == "snn_interpreter.hub.cli:main"
    assert scripts["snn-energy"] == "snn_interpreter.energy.cli:main"
    assert scripts["snn-benchmark"] == "snn_interpreter.benchmark.cli:main"


def test_core_excludes_the_server_root() -> None:
    """Core discovery excludes the server root so it cannot leak."""
    find = _pyproject(_CORE)["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["snn_interpreter*"]
    assert "server*" in find["exclude"]


def test_server_declares_base_dependencies() -> None:
    """The server promotes the old web extra to base dependencies."""
    deps = set(_pyproject(_SERVER)["project"]["dependencies"])
    assert deps >= _EXPECTED_SERVER_DEPS
    assert any(dep.startswith("snn-interpreter~=") for dep in deps)


def test_server_owns_only_the_server_root() -> None:
    """Server discovery is independent of the core package."""
    find = _pyproject(_SERVER)["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["server*"]
    assert "snn_interpreter*" in find["exclude"]


def _compose() -> Dict[str, Any]:
    """Parse ``docker-compose.yml`` with the optional PyYAML dependency."""
    yaml = pytest.importorskip("yaml")
    text = (_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def test_compose_default_service_is_profile_free() -> None:
    """``docker compose up --build`` still starts only the default service."""
    service = _compose()["services"]["snn-interpreter"]
    assert "profiles" not in service
    # The host port is overridable via SNN_HOST_PORT, but the container port
    # stays 8877 so the single-port dashboard contract is unchanged.
    assert any(port.endswith(":8877") for port in service["ports"])


def test_compose_declares_cpu_and_gpu_profiles() -> None:
    """The cpu and gpu profiles are declared by the alternate services."""
    services = _compose()["services"]
    profiles: Set[str] = set()
    for service in services.values():
        profiles.update(service.get("profiles", []))
    assert {"cpu", "gpu"} <= profiles


def test_compose_text_mentions_profiles() -> None:
    """A dependency-free check keeps the profile contract covered."""
    text = (_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "profiles:" in text
    assert "cpu" in text
    assert "gpu" in text
