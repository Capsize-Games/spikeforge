"""Refuse image loading when Docker's actual filesystem lacks capacity."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def docker_root() -> Path:
    """Resolve the actual Docker storage path instead of assuming root."""
    result = subprocess.run(
        ["docker", "info", "--format", "{{.DockerRootDir}}"], check=True,
        text=True, capture_output=True,
    )
    return Path(result.stdout.strip())


def require_capacity(size: int, root: Path) -> None:
    """Compare the candidate's unpacked image bytes to available storage."""
    if size <= 0:
        raise ValueError("Image size must be positive")
    free = shutil.disk_usage(root).free
    if free < size:
        raise ValueError(
            f"Insufficient Docker storage: image={size} bytes, "
            f"available={free} bytes on {root}"
        )
    print(f"Docker storage {root}: image={size}, available={free} bytes")


if __name__ == "__main__":
    storage = Path(sys.argv[2]) if len(sys.argv) == 3 else docker_root()
    require_capacity(int(sys.argv[1]), storage)
