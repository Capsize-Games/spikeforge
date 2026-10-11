"""Owned real Docker/Compose resources for deployment boundary tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
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
    try:
        path = compose_file(directory, tag)
        yield revision, project, path
    finally:
        path = directory / "docker-compose.hetzner.yml"
        try:
            if path.exists():
                docker("compose", "-p", project, "-f", str(path),
                       "down", "--volumes", "--remove-orphans")
        finally:
            docker("image", "rm", tag)


def run_script(name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the production helper without substitutions or production SSH."""
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / name), *arguments],
        text=True, capture_output=True, timeout=120,
    )


def saved_config_digest(tag: str, path: Path) -> str:
    """Hash the actual archived configuration using independent tar access."""
    docker("image", "save", "--output", str(path), tag)
    with tarfile.open(path) as archive:
        manifest = archive.extractfile("manifest.json")
        assert manifest is not None
        with manifest:
            config = json.load(manifest)[0]["Config"]
        payload = archive.extractfile(config)
        assert payload is not None
        with payload:
            return "sha256:" + hashlib.sha256(payload.read()).hexdigest()
