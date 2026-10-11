"""Protect shared live edge routes during concurrent docs installation."""

from __future__ import annotations

import fcntl
import subprocess
from pathlib import Path

import pytest
from docs_edge_process import (
    config,
    docker,
    docs_process,
    edge,
    finish,
    publish,
    response,
)


def require_blocked(process: subprocess.Popen[str]) -> None:
    """Require the second installer to block while the owner holds flock."""
    try:
        output, _ = process.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        return
    pytest.fail(
        "Docs installer finished while Caddyfile.lock was held; "
        f"exit={process.returncode}\n{output}"
    )


def test_docs_waits_and_reloads_latest_shared_routes(tmp_path: Path) -> None:
    """Wait for the shared owner and retain its new live and disk routes."""
    with edge(tmp_path) as owner:
        _, base, path = owner
        with path.with_name("Caddyfile.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with docs_process() as process:
                require_blocked(process)
                publish(owner, config(hub=True))
                fcntl.flock(lock, fcntl.LOCK_UN)
                finish(process)
        assert path.read_text() == config(hub=True)
        assert response(base, "hub.spikeforge.net") == "hub"
        assert response(base, "docs.spikeforge.net") == "docs-before"
        assert response(base, "other.example") == "other"


def test_docs_reloads_host_snapshot_with_stale_bind(tmp_path: Path) -> None:
    """Keep explicit candidate reload after a host inode is replaced."""
    with edge(tmp_path) as (name, base, path):
        replacement = path.with_name("replacement")
        replacement.write_text(config("docs-after", hub=True))
        replacement.replace(path)
        assert "docs-before" in docker(
            "exec", name, "cat", "/etc/caddy/Caddyfile"
        )
        with docs_process() as process:
            finish(process)
        assert response(base, "hub.spikeforge.net") == "hub"
        assert response(base, "docs.spikeforge.net") == "docs-after"
        assert response(base, "other.example") == "other"
