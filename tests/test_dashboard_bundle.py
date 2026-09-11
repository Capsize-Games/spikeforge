"""Pinned dashboard bundle resolution (no browser required).

``SPIKEFORGE_DASHBOARD_DIST`` lets the server serve a prebuilt ``dist/``
published by ``capsize-games/spikeforge-dashboard`` instead of rebuilding
the in-repo ``client/`` mirror. The fallback to today's ``client/dist``
behaviour must be preserved.
"""

from pathlib import Path

import pytest

from server import web
from spikeforge import config


def _absent(tmp_path: Path, name: str) -> Path:
    """Return a path that deliberately does not exist."""
    return tmp_path / name


def test_a_pinned_bundle_is_preferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When SPIKEFORGE_DASHBOARD_DIST is present it wins over client/dist."""
    dist = tmp_path / "dist"
    dist.mkdir()
    monkeypatch.setattr(config, "DASHBOARD_DIST", str(dist))
    monkeypatch.setattr(web, "_CLIENT_DIST", _absent(tmp_path, "docker"))
    monkeypatch.setattr(web, "_LOCAL_DIST", _absent(tmp_path, "local"))
    assert web.pinned_dist() == dist
    assert web.client_dist() == dist


def test_an_absent_pin_falls_back_to_client_dist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured-but-missing path keeps today's client/dist behaviour."""
    local = tmp_path / "client-dist"
    local.mkdir()
    monkeypatch.setattr(config, "DASHBOARD_DIST", str(tmp_path / "absent"))
    monkeypatch.setattr(web, "_CLIENT_DIST", _absent(tmp_path, "docker"))
    monkeypatch.setattr(web, "_LOCAL_DIST", local)
    assert web.pinned_dist() is None
    assert web.client_dist() == local


def test_an_unset_pin_keeps_the_legacy_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no pin, the Docker path is preferred, then the in-repo one."""
    docker = tmp_path / "docker-dist"
    docker.mkdir()
    monkeypatch.setattr(config, "DASHBOARD_DIST", None)
    monkeypatch.setattr(web, "_CLIENT_DIST", docker)
    monkeypatch.setattr(web, "_LOCAL_DIST", _absent(tmp_path, "local"))
    assert web.pinned_dist() is None
    assert web.client_dist() == docker
