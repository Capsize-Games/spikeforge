"""Core, server, and targets packaging: extras, scripts, and pins."""

from _packaging_helpers import (
    _CORE,
    _EXPECTED_CORE_DEPS,
    _EXPECTED_EXTRAS,
    _EXPECTED_SCRIPTS,
    _EXPECTED_SERVER_DEPS,
    _EXPECTED_SERVER_SCRIPTS,
    _EXPECTED_TARGETS_DEPS,
    _EXPECTED_TARGETS_EXTRAS,
    _EXPECTED_TARGETS_SCRIPTS,
    _HUB,
    _SERVER,
    _TARGETS,
    _core_pin,
    _pyproject,
    _satellite_pin,
)


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
        _satellite_pin(_TARGETS),
        _satellite_pin(_HUB),
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
    assert _satellite_pin(_TARGETS) in deps
    assert _satellite_pin(_HUB) in deps
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
    assert _core_pin() in deps


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
