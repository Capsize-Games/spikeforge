"""Packaging extras/console scripts and docker-compose profile config."""

from pathlib import Path
from typing import Any, Dict, Set

import pytest

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

_ROOT = Path(__file__).resolve().parent.parent
_CORE = _ROOT / "packages" / "spikeforge" / "pyproject.toml"
_SERVER = _ROOT / "packages" / "spikeforge-server" / "pyproject.toml"
_TARGETS = _ROOT / "packages" / "spikeforge-targets" / "pyproject.toml"
_HUB = _ROOT / "packages" / "spikeforge-hub" / "pyproject.toml"

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
    "spikeforge-targets~=0.1.0",
    "spikeforge-hub~=0.1.0",
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


def test_core_all_extra_bundles_the_satellites() -> None:
    """The ``all`` extra pulls the targets and hub satellites only."""
    extras = _pyproject(_CORE)["project"]["optional-dependencies"]
    assert set(extras["all"]) == {
        "spikeforge-targets~=0.1.0",
        "spikeforge-hub~=0.1.0",
    }
    # The base install stays clean: no satellite may be a hard dependency.
    assert not any(
        dep.startswith(("spikeforge-targets", "spikeforge-hub"))
        for dep in _pyproject(_CORE)["project"]["dependencies"]
    )


def test_console_script_targets_are_stable() -> None:
    """The five remaining core entry points resolve to their targets."""
    scripts = _pyproject(_CORE)["project"]["scripts"]
    assert scripts["spikeforge"] == "main:main"
    assert scripts["spikeforge-encodings"] == "main_encodings:main"
    assert scripts["spikeforge-verify"] == "spikeforge.cli.verify:main"
    assert scripts["spikeforge-records"] == "spikeforge.cli.records_cli:main"
    assert scripts["spikeforge-benchmark"] == "spikeforge.benchmark.cli:main"


def test_core_excludes_the_server_root() -> None:
    """Core discovery excludes the satellite roots so they cannot leak."""
    find = _pyproject(_CORE)["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["spikeforge", "spikeforge.*"]
    assert "server*" in find["exclude"]
    assert "spikeforge_targets*" in find["exclude"]
    assert "spikeforge_hub*" in find["exclude"]


def test_server_declares_base_dependencies() -> None:
    """The server promotes the old web extra to base dependencies."""
    deps = set(_pyproject(_SERVER)["project"]["dependencies"])
    assert deps >= _EXPECTED_SERVER_DEPS
    assert any(dep.startswith("spikeforge~=") for dep in deps)


def test_server_console_script_resolves_to_main() -> None:
    """``spikeforge-server`` starts the uvicorn app via ``__main__:main``."""
    scripts = _pyproject(_SERVER)["project"]["scripts"]
    assert set(scripts) == _EXPECTED_SERVER_SCRIPTS
    assert scripts["spikeforge-server"] == "server.__main__:main"


def test_server_owns_only_the_server_root() -> None:
    """Server discovery is independent of the core package."""
    find = _pyproject(_SERVER)["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["server*"]
    assert "spikeforge*" in find["exclude"]


def test_targets_distribution_owns_the_moved_surface() -> None:
    """spikeforge-targets owns the moved extras, scripts, and import root."""
    project = _pyproject(_TARGETS)["project"]
    assert set(project["optional-dependencies"]) == _EXPECTED_TARGETS_EXTRAS
    assert set(project["scripts"]) == _EXPECTED_TARGETS_SCRIPTS
    deps = set(project["dependencies"])
    assert deps >= _EXPECTED_TARGETS_DEPS
    assert "spikeforge~=0.3.1" in deps


def test_targets_console_script_targets_resolve_to_the_new_root() -> None:
    """Moved console scripts resolve inside the ``spikeforge_targets`` root."""
    scripts = _pyproject(_TARGETS)["project"]["scripts"]
    assert scripts["spikeforge-energy"] == "spikeforge_targets.energy.cli:main"
    assert scripts["spikeforge-targets"] == (
        "spikeforge_targets.cli.target_cli:main"
    )


def test_targets_owns_only_the_targets_root() -> None:
    """Targets discovery is independent of core and the server."""
    find = _pyproject(_TARGETS)["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["spikeforge_targets*"]
    assert "spikeforge" in find["exclude"]
    assert "server*" in find["exclude"]


def test_hub_distribution_owns_the_moved_surface() -> None:
    """spikeforge-hub owns the moved extras, script, and import root."""
    project = _pyproject(_HUB)["project"]
    assert set(project["optional-dependencies"]) == _EXPECTED_HUB_EXTRAS
    assert set(project["scripts"]) == _EXPECTED_HUB_SCRIPTS
    deps = set(project["dependencies"])
    assert deps >= _EXPECTED_HUB_DEPS
    assert "spikeforge~=0.3.1" in deps


def test_hub_console_script_targets_resolve_to_the_new_root() -> None:
    """The hub's own scripts resolve inside the ``spikeforge_hub`` root."""
    scripts = _pyproject(_HUB)["project"]["scripts"]
    assert scripts["spikeforge-hub"] == "spikeforge_hub.cli:main"
    # PT-W7: the registry CLI is owned by the hub distribution.
    assert scripts["spikeforge-registry"] == "spikeforge_hub.registry_cli:main"


def test_hub_owns_only_the_hub_root() -> None:
    """Hub discovery is independent of core, targets, and the server."""
    find = _pyproject(_HUB)["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["spikeforge_hub*"]
    assert "spikeforge" in find["exclude"]
    assert "server*" in find["exclude"]
    assert "spikeforge_targets*" in find["exclude"]


def test_clients_distribution_owns_the_client_surface() -> None:
    """The clients distribution declares its script and no core runtime."""
    project = _pyproject(_CLIENTS)["project"]
    assert set(project["scripts"]) == _EXPECTED_CLIENTS_SCRIPTS
    # A client install must never pull the torch-backed core.
    assert project["dependencies"] == []
    assert "spikeforge~=0.3.1" not in set(project["dependencies"])


def test_clients_console_script_resolves_to_the_new_root() -> None:
    """The client CLI resolves inside the ``spikeforge_clients`` root."""
    scripts = _pyproject(_CLIENTS)["project"]["scripts"]
    assert scripts["spikeforge-clients"] == "spikeforge_clients.cli:main"


def test_clients_owns_only_the_clients_root() -> None:
    """Clients discovery is independent of every other distribution."""
    find = _pyproject(_CLIENTS)["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["spikeforge_clients*"]
    for other in (
        "spikeforge",
        "spikeforge.*",
        "server*",
        "tests*",
        "spikeforge_targets*",
        "spikeforge_hub*",
        "spikeforge_serve*",
    ):
        assert other in find["exclude"]


def test_io_distribution_owns_the_io_surface() -> None:
    """PT-W7's spikeforge-io owns one script, one root, and a core pin."""
    project = _pyproject(_IO)["project"]
    assert set(project["scripts"]) == _EXPECTED_IO_SCRIPTS
    deps = set(project["dependencies"])
    assert deps >= _EXPECTED_IO_DEPS
    assert "spikeforge~=0.3.1" in deps
    assert project["version"] == "0.1.0"


def test_io_console_script_resolves_to_the_new_root() -> None:
    """The I/O CLI resolves inside the ``spikeforge_io`` root."""
    scripts = _pyproject(_IO)["project"]["scripts"]
    assert scripts["spikeforge-io"] == "spikeforge_io.cli:main"


def test_io_owns_only_the_io_root() -> None:
    """I/O discovery is independent of every other distribution."""
    find = _pyproject(_IO)["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["spikeforge_io*"]
    for other in (
        "spikeforge",
        "spikeforge.*",
        "server*",
        "tests*",
        "spikeforge_targets*",
        "spikeforge_hub*",
        "spikeforge_serve*",
        "spikeforge_clients*",
    ):
        assert other in find["exclude"]


def _compose() -> Dict[str, Any]:
    """Parse ``docker-compose.yml`` with the optional PyYAML dependency."""
    yaml = pytest.importorskip("yaml")
    text = (_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def test_compose_default_service_is_profile_free() -> None:
    """``docker compose up --build`` still starts only the default service."""
    service = _compose()["services"]["spikeforge"]
    assert "profiles" not in service
    # The host port is overridable via SPIKEFORGE_HOST_PORT, but the
    # container port stays 8877 so the single-port dashboard contract is
    # unchanged.
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
