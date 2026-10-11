"""Keep dependency builds outside the shared production host."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/deploy-hetzner.yml"


def test_remote_deployment_cannot_build_dependencies() -> None:
    """The actual SSH workflow must load and start a prebuilt image."""
    workflow = WORKFLOW.read_text()
    assert "up -d --build" not in workflow
    assert 'docker image save "$image"' in workflow
    assert "docker image load" in workflow
    assert "scripts/start-dashboard-image.sh" in workflow


def test_hosted_build_precedes_the_private_network() -> None:
    """The CPU image build finishes before the host becomes accessible."""
    workflow = WORKFLOW.read_text()
    assert "runs-on: ubuntu-latest" in workflow
    assert workflow.index("scripts/build-dashboard-image.sh") < (
        workflow.index("Connect to the private deployment network")
    )


def test_only_current_main_can_mutate_the_host() -> None:
    """Manual and queued deployments preserve the current main revision."""
    workflow = WORKFLOW.read_text()
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert workflow.count("scripts/check-dashboard-source.sh") >= 2
    assert "cancel-in-progress: false" in workflow


def test_revision_image_is_persisted_and_context_omits_credentials() -> None:
    """Future Compose commands select the loaded image without secrets."""
    workflow = WORKFLOW.read_text()
    assert "scripts/write-dashboard-env.py" in workflow
    assert "SPIKEFORGE_DASHBOARD_IMAGE" in (
        ROOT / "docker-compose.hetzner.yml"
    ).read_text()
    ignored = (ROOT / ".dockerignore").read_text().splitlines()
    assert ".env" in ignored
    assert ".venv/" in ignored
