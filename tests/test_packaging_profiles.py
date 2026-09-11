"""Packaging extras/console scripts and docker-compose profile config."""

from pathlib import Path
from typing import Any, Dict, Set

import pytest

_ROOT = Path(__file__).resolve().parent.parent

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
_EXPECTED_EXTRAS = {
    "dev",
    "web",
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


def _setup_source() -> str:
    """Return the text of the project's ``setup.py``."""
    return (_ROOT / "setup.py").read_text(encoding="utf-8")


def test_setup_declares_expected_extras_and_scripts() -> None:
    """Every documented extra and console script is exported."""
    source = _setup_source()
    for extra in _EXPECTED_EXTRAS:
        assert f'"{extra}":' in source
    for script in _EXPECTED_SCRIPTS:
        assert script in source


def _compose() -> Dict[str, Any]:
    """Parse ``docker-compose.yml`` with the optional PyYAML dependency."""
    yaml = pytest.importorskip("yaml")
    text = (_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def test_compose_default_service_is_profile_free() -> None:
    """``docker compose up --build`` still starts only the default service."""
    service = _compose()["services"]["snn-interpreter"]
    assert "profiles" not in service
    assert "8877:8877" in service["ports"]


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
