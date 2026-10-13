"""Keep production secrets while persisting the immutable image choice."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_environment_replaces_stale_image_and_preserves_secrets(
    tmp_path: Path,
) -> None:
    """The actual stdin writer atomically installs a private environment."""
    path = tmp_path / ".env"
    revision = "a" * 40
    body = ("SECRET=keep-this\nSPIKEFORGE_DASHBOARD_IMAGE=old\n"
            "export SPIKEFORGE_DASHBOARD_IMAGE=older\n"
            "SPIKEFORGE_DASHBOARD_IMAGE = spaced\n"
            "export  SPIKEFORGE_DASHBOARD_IMAGE=exported\n")
    result = subprocess.run(
        [os.sys.executable, str(ROOT / "scripts/write-dashboard-env.py"),
         str(path), revision], input=body, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert path.read_text() == (
        "SECRET=keep-this\n"
        f"SPIKEFORGE_DASHBOARD_IMAGE=spikeforge-dashboard:{revision}\n"
    )
    assert path.stat().st_mode & 0o777 == 0o600


def test_invalid_revision_cannot_replace_environment(tmp_path: Path) -> None:
    """Malformed revision input fails before writing or exposing secrets."""
    path = tmp_path / ".env"
    path.write_text("SECRET=existing\n")
    result = subprocess.run(
        [os.sys.executable, str(ROOT / "scripts/write-dashboard-env.py"),
         str(path), "not-a-revision"], input="SECRET=new\n", text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Invalid deployment revision" in result.stderr
    assert path.read_text() == "SECRET=existing\n"
