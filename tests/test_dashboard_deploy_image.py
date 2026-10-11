"""Require image-only deployment using actual isolated Docker/Compose."""

from pathlib import Path

from dashboard_deploy_docker import docker, resources, run_script


def test_missing_image_fails_before_compose_can_build(tmp_path: Path) -> None:
    """An absent revision cannot trigger a dependency build or a start."""
    with resources(tmp_path) as (_, project, path):
        absent = "0" * 40
        result = run_script(
            "start-dashboard-image.sh", absent, str(tmp_path), project,
        )
        assert result.returncode != 0
        assert "Required dashboard image is not loaded" in result.stderr
        assert not docker("compose", "-p", project, "-f", str(path), "ps",
                          "--all", "-q")


def test_preloaded_image_starts_with_verified_identity(tmp_path: Path) -> None:
    """Compose starts the loaded image and verifies its actual container."""
    with resources(tmp_path) as (revision, project, path):
        result = run_script(
            "start-dashboard-image.sh", revision, str(tmp_path), project,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        container = docker("compose", "-p", project, "-f", str(path),
                           "ps", "-q", "spikeforge-dashboard")
        check = run_script("check-dashboard-revision.sh", revision, container)
        assert check.returncode == 0, check.stderr


def test_wrong_running_revision_is_rejected(tmp_path: Path) -> None:
    """A real container with a different OCI revision is never accepted."""
    with resources(tmp_path) as (_, project, path):
        docker("compose", "-p", project, "-f", str(path), "up", "-d",
               "--no-build", "--pull", "never", "--wait")
        container = docker("compose", "-p", project, "-f", str(path),
                           "ps", "-q", "spikeforge-dashboard")
        result = run_script("check-dashboard-revision.sh", "0" * 40,
                            container)
        assert result.returncode != 0
        assert "Dashboard revision mismatch" in result.stderr
