"""Satellite packaging: hub, clients, and I/O distributions."""

from _packaging_helpers import (
    _CLIENTS,
    _EXPECTED_CLIENTS_SCRIPTS,
    _EXPECTED_HUB_DEPS,
    _EXPECTED_HUB_EXTRAS,
    _EXPECTED_HUB_SCRIPTS,
    _EXPECTED_IO_DEPS,
    _EXPECTED_IO_SCRIPTS,
    _HUB,
    _IO,
    _core_pin,
    _matrix_version,
    _pyproject,
)


def test_hub_distribution_owns_the_moved_surface() -> None:
    """spikeforge-hub owns the moved extras, script, and import root."""
    project = _pyproject(_HUB)["project"]
    assert set(project["optional-dependencies"]) == _EXPECTED_HUB_EXTRAS
    assert set(project["scripts"]) == _EXPECTED_HUB_SCRIPTS
    deps = set(project["dependencies"])
    assert deps >= _EXPECTED_HUB_DEPS
    assert _core_pin() in deps


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
    assert _core_pin() in deps
    assert project["version"] == _matrix_version("spikeforge-io")


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
