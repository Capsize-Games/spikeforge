"""Exercise the queued-revision guard with an actual local Git remote."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def git(directory: Path, *arguments: str) -> str:
    """Run actual Git, keeping this test separate from project branches."""
    return subprocess.run(
        ["git", "-C", str(directory), *arguments], check=True, text=True,
        capture_output=True,
    ).stdout.strip()


def repository(path: Path) -> tuple[str, str]:
    """Create a bare origin and two revisions of its main branch."""
    git(path, "init", "--bare", "remote.git")
    git(path, "init", "--initial-branch=main", "source")
    source = path / "source"
    git(source, "config", "user.email", "deployment-test@example.invalid")
    git(source, "config", "user.name", "Deployment test fixture")
    git(source, "remote", "add", "origin", str(path / "remote.git"))
    git(source, "commit", "--allow-empty", "-m", "first")
    first = git(source, "rev-parse", "HEAD")
    git(source, "commit", "--allow-empty", "-m", "current")
    current = git(source, "rev-parse", "HEAD")
    git(source, "push", "origin", "main")
    return first, current


@pytest.mark.parametrize("stale", [False, True])
def test_only_remote_main_can_deploy(tmp_path: Path, stale: bool) -> None:
    """The real guard checks ls-remote, rather than local cached refs."""
    first, current = repository(tmp_path)
    revision = first if stale else current
    source = tmp_path / "source"
    git(source, "checkout", "--detach", revision)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/check-dashboard-source.sh")],
        cwd=source, env={**os.environ, "GITHUB_SHA": revision},
        text=True, capture_output=True,
    )
    assert result.returncode == (1 if stale else 0), result.stderr
    if stale:
        assert "superseded" in result.stderr
