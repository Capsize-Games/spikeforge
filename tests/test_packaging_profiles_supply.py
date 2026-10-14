"""Supply chain: compose profiles, install closure, and lockfiles."""

import re
from typing import Any, Dict, Set

import pytest
from _packaging_helpers import _ROOT, _pyproject


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


@pytest.mark.parametrize(
    "relative",
    ["Dockerfile", ".github/workflows/ci.yml", "install.sh", "scripts/dev.sh"],
)
def test_install_lists_cover_the_server_dependency_closure(
    relative: str,
) -> None:
    """Anything installing spikeforge-server installs what it depends on.

    These four files install the server chain from ``./packages/*``. Omitting
    one of its local dependencies does not fail loudly: pip resolves that pin
    from PyPI instead, so the build silently mixes this checkout with the last
    published wheel -- until the pin names a version PyPI does not have, and
    then every job fails at once. Both happened: the deployed dashboard ran a
    published ``spikeforge-serve`` for several releases, and the 0.2.1 bump
    broke the image build and all of CI.

    The check is per file rather than per command, which is coarse but keeps
    it readable; build-only references are excluded so a ``python -m build``
    elsewhere in the file cannot satisfy it.
    """
    text = (_ROOT / relative).read_text(encoding="utf-8")
    installed: Set[str] = set()
    for line in text.splitlines():
        if "python -m build" in line or "dist/" in line or "-wheel " in line:
            continue
        installed.update(re.findall(r"\./packages/([\w-]+)", line))
    if "spikeforge-server" not in installed:
        pytest.skip(f"{relative} does not install spikeforge-server")

    pending = ["spikeforge-server"]
    required: Set[str] = set()
    while pending:
        name = pending.pop()
        if name in required:
            continue
        required.add(name)
        path = _ROOT / "packages" / name / "pyproject.toml"
        for dep in _pyproject(path)["project"].get("dependencies", []):
            dist = re.split(r"[<>=~!\[; ]", dep, maxsplit=1)[0]
            if dist.startswith("spikeforge"):
                pending.append(dist)

    assert required <= installed, (
        f"{relative} omits {sorted(required - installed)}; pip would resolve "
        f"those from PyPI instead of this checkout"
    )


_LOCKS = {
    "requirements/server.cpu.lock": "https://download.pytorch.org/whl/cpu",
    "requirements/server.cu132.lock": (
        "https://download.pytorch.org/whl/cu132"
    ),
}


def _locked_versions(relative: str) -> Dict[str, str]:
    """Return the ``{name: version}`` pins locked in ``relative``."""
    versions: Dict[str, str] = {}
    for line in (_ROOT / relative).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^ ]+)", line)
        if match:
            versions[match.group(1).lower()] = match.group(2)
    return versions


def _public(version: str) -> str:
    """Return ``version`` without its torch ``+cpu``/``+cu132`` segment."""
    return version.split("+", maxsplit=1)[0]


def test_dockerfile_installs_from_a_hashed_lockfile() -> None:
    """The image installs the closure with hash verification enabled."""
    text = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "--require-hashes" in text
    assert re.search(r"-r \$\{LOCKFILE\}", text)
    assert "pip install --index-url" not in text
    assert "pip install -r requirements.txt" not in text


def test_dockerfile_local_installs_fail_closed() -> None:
    """A missing local dist fails the build instead of using PyPI."""
    text = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "--no-deps ./packages/" in text
    assert "pip check" in text


def test_dockerfile_records_the_lockfile_cadence() -> None:
    """The Dockerfile header says when the locks are recompiled."""
    text = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "Lockfile cadence" in text


def test_lockfiles_carry_hashes_for_every_pin() -> None:
    """Every locked pin carries hashes, so installs verify bytes."""
    for relative, index in _LOCKS.items():
        text = (_ROOT / relative).read_text(encoding="utf-8")
        assert f"--index-url {index}\n" in text
        pins = _locked_versions(relative)
        assert pins, f"{relative} locks nothing"
        assert len(re.findall(r"--hash=sha256:", text)) >= len(pins)


def test_lockfiles_agree_on_shared_versions() -> None:
    """The CPU and CUDA locks differ only by torch build and extras."""
    versions = {
        relative: _locked_versions(relative) for relative in _LOCKS
    }
    cpu, cuda = (
        versions["requirements/server.cpu.lock"],
        versions["requirements/server.cu132.lock"],
    )
    shared = set(cpu) & set(cuda)
    assert shared, "the locks share no packages"
    mismatched = {
        name
        for name in shared
        if _public(cpu[name]) != _public(cuda[name])
    }
    assert not mismatched, f"version drift across locks: {mismatched}"
    assert cpu["torch"].endswith("+cpu")
    assert cuda["torch"].endswith("+cu132")


def test_requirements_match_the_lockfiles() -> None:
    """requirements.txt pins the same public versions as the locks."""
    wanted: Dict[str, str] = {}
    for line in (_ROOT / "requirements.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, version = stripped.partition("==")
        assert separator, f"floating pin in requirements.txt: {stripped!r}"
        wanted[name.strip().lower()] = version.strip()
    assert wanted, "requirements.txt pins nothing"
    for relative in _LOCKS:
        locked = _locked_versions(relative)
        for name, version in wanted.items():
            assert name in locked, f"{name} is missing from {relative}"
            assert _public(locked[name]) == version, (
                f"{name}=={version} disagrees with {relative} "
                f"({locked[name]})"
            )
