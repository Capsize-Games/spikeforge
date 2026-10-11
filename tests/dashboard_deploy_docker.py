"""Owned real Docker/Compose resources for deployment boundary tests."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "org.opencontainers.image.revision"


def docker(*arguments: str) -> str:
    """Execute Docker itself and retain diagnostics for a failed operation."""
    return subprocess.run(
        ["docker", *arguments], text=True, check=True,
        capture_output=True, timeout=120,
    ).stdout.strip()


def image(directory: Path, revision: str) -> str:
    """Build a real small image with an explicit immutable revision label."""
    tag = f"spikeforge-dashboard:{revision}"
    context = directory / revision
    context.mkdir()
    (context / "Dockerfile").write_text(
        "FROM python:3.13-slim\n"
        f"LABEL {REVISION}={revision}\n"
        'CMD ["sleep", "300"]\n'
    )
    docker("build", "-q", "-t", tag, str(context))
    return tag


def compose_file(directory: Path, tag: str) -> Path:
    """Isolate the actual production model from host data and networking."""
    environment = {**os.environ, "SPIKEFORGE_DASHBOARD_TOKEN": "local-test",
                   "SPIKEFORGE_DASHBOARD_IMAGE": tag}
    result = subprocess.run(
        ["docker", "compose", "-f", str(ROOT / "docker-compose.hetzner.yml"),
         "config", "--format", "json"], env=environment,
        check=True, text=True, capture_output=True,
    )
    model = json.loads(result.stdout)
    service = model["services"]["spikeforge-dashboard"]
    service.pop("container_name")
    service.update(image=tag, volumes=[], networks=["default"], restart="no")
    service["healthcheck"] = {"test": ["CMD", "python", "-c", "pass"],
                              "interval": "1s", "timeout": "1s"}
    model.update(networks={"default": {}}, volumes={})
    path = directory / "docker-compose.hetzner.yml"
    path.write_text(json.dumps(model))
    (directory / ".env").write_text("SPIKEFORGE_DASHBOARD_TOKEN=local-test\n")
    return path


@contextmanager
def resources(directory: Path) -> Iterator[tuple[str, str, Path]]:
    """Create and clean only a unique local Compose project and image."""
    revision = uuid.uuid4().hex + uuid.uuid4().hex[:8]
    project = f"dashboard-deploy-{uuid.uuid4().hex}"
    tag = image(directory, revision)
    path = compose_file(directory, tag)
    try:
        yield revision, project, path
    finally:
        docker("compose", "-p", project, "-f", str(path),
               "down", "--volumes", "--remove-orphans")
        docker("image", "rm", tag)


def run_script(name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the production helper without substitutions or production SSH."""
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / name), *arguments],
        text=True, capture_output=True, timeout=120,
    )
