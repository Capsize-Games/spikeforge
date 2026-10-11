"""Check real filesystem capacity before loading the deployment image."""

import subprocess
import sys
from pathlib import Path
from shutil import disk_usage

ROOT = Path(__file__).resolve().parents[1]


def check(size: int, root: Path) -> subprocess.CompletedProcess[str]:
    """Execute the guard against an actual filesystem without overrides."""
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-dashboard-capacity.py"),
         str(size), str(root)], text=True, capture_output=True,
    )


def test_an_image_larger_than_filesystem_capacity_is_rejected(
    tmp_path: Path,
) -> None:
    """Fail closed before load without writing a large test image."""
    result = check(disk_usage(tmp_path).total + 1, tmp_path)
    assert result.returncode != 0
    assert "Insufficient Docker storage" in result.stderr


def test_small_image_with_real_free_space_is_accepted(tmp_path: Path) -> None:
    """Actual available capacity permits a one-byte image requirement."""
    assert disk_usage(tmp_path).free > 1
    result = check(1, tmp_path)
    assert result.returncode == 0, result.stderr


def test_invalid_image_size_fails_closed(tmp_path: Path) -> None:
    """Invalid size cannot make a capacity check silently succeed."""
    result = check(-1, tmp_path)
    assert result.returncode != 0
    assert "Image size must be positive" in result.stderr
