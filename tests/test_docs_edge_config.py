"""Checks for the shared Caddy reload used by the docs deployment."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy/configure-docs.sh"


def _fake_docker(directory: Path) -> Path:
    docker = directory / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  ps) echo fake-caddy ;;\n"
        '  inspect) echo "$MOCK_CADDYFILE" ;;\n'
        '  cp) cp "$2" "$MOCK_CANDIDATE" ;;\n'
        '  exec) printf "%s\\n" "$*" >> "$MOCK_LOG" ;;\n'
        "esac\n"
    )
    docker.chmod(0o755)
    return docker


def test_docs_reload_uses_current_host_file(tmp_path: Path) -> None:
    """Validate and reload the host file even with a stale bind mount."""
    config = tmp_path / "Caddyfile"
    config.write_text(
        "(cf_origin_only) { respond 403 }\n"
        "docs.spikeforge.net { import cf_origin_only }\n"
    )
    candidate = tmp_path / "candidate"
    log = tmp_path / "docker.log"
    _fake_docker(tmp_path)
    env = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "MOCK_CADDYFILE": str(config),
        "MOCK_CANDIDATE": str(candidate),
        "MOCK_LOG": str(log),
    }
    subprocess.run(["bash", str(SCRIPT)], check=True, env=env)
    assert candidate.read_text() == config.read_text()
    calls = log.read_text().splitlines()
    assert "caddy validate --config /tmp/spikeforge-docs-candidate" in calls[0]
    assert "caddy reload --config /tmp/spikeforge-docs-candidate" in calls[1]
