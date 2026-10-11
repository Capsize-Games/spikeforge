"""Real Docker/Caddy processes for the shared docs edge contract."""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy/configure-docs.sh"
LABEL = "com.docker.compose.service=caddy"
Edge = tuple[str, str, Path]


def docker(*arguments: str) -> str:
    """Run actual Docker commands, retaining failure diagnostics."""
    return subprocess.run(
        ["docker", *arguments], check=True, text=True,
        capture_output=True, timeout=60,
    ).stdout.strip()


def config(docs: str = "docs-before", *, hub: bool = False) -> str:
    """Build plaintext local routes without requesting public certificates."""
    routes = {"docs.spikeforge.net": docs, "other.example": "other"}
    if hub:
        routes["hub.spikeforge.net"] = "hub"
    return "\n".join(
        f'http://{host}:8080 {{\n respond "{body}"\n}}\n'
        for host, body in routes.items()
    )


def response(base: str, host: str) -> str:
    """Request a host route from the actual local Caddy server."""
    request = urllib.request.Request(base, headers={"Host": host})
    with urllib.request.urlopen(request, timeout=2) as result:
        return result.read().decode()


def wait_ready(base: str) -> None:
    """Wait for the container's HTTP listener to accept real requests."""
    for _ in range(100):
        try:
            if response(base, "docs.spikeforge.net") == "docs-before":
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    raise AssertionError("Disposable Caddy did not become ready")


def start_edge(path: Path) -> tuple[str, str]:
    """Fail closed before exposing the legacy script to another Caddy."""
    assert not docker("ps", "-a", "--filter", f"label={LABEL}", "-q"), (
        "An unrelated compose-service=caddy container already exists"
    )
    name = f"spikeforge-docs-test-{uuid.uuid4().hex}"
    docker("run", "-d", "--name", name, "--label", LABEL,
           "-v", f"{path}:/etc/caddy/Caddyfile:ro",
           "-p", "127.0.0.1::8080", "caddy:2.10.2-alpine")
    address = docker("port", name, "8080/tcp")
    return name, f"http://{address}/"


@contextmanager
def edge(directory: Path) -> Iterator[Edge]:
    """Own a disposable edge and remove it even after an assertion fails."""
    path = directory / "Caddyfile"
    path.write_text(config())
    name, base = start_edge(path)
    try:
        assert docker("ps", "--filter", f"label={LABEL}",
                      "--format", "{{.Names}}") == name
        wait_ready(base)
        yield name, base, path
    finally:
        docker("rm", "-f", name)


@contextmanager
def docs_process() -> Iterator[subprocess.Popen[str]]:
    """Run the actual installer in an owned, cancellable process group."""
    process = subprocess.Popen(
        ["bash", str(SCRIPT)], text=True, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=10)


def finish(process: subprocess.Popen[str]) -> None:
    """Require the installer to complete successfully with real Caddy."""
    output, _ = process.communicate(timeout=30)
    assert process.returncode == 0, output


def publish(owner: Edge, content: str) -> None:
    """Act as the current lock owner, publishing a fresh route snapshot."""
    name, _, path = owner
    path.write_text(content)
    docker("cp", str(path), f"{name}:/tmp/owner-candidate")
    docker("exec", name, "caddy", "reload", "--config",
           "/tmp/owner-candidate", "--adapter", "caddyfile")
