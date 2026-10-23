"""Basic cross-process locking checks for the pinned filelock dependency."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import filelock
import pytest
from filelock import ReadWriteLock, SoftFileLock


def _child_attempt(path: Path, lock_kind: str) -> str:
    script = """\
import sys
import filelock
from filelock import ReadWriteLock, SoftFileLock

path, kind = sys.argv[1:]
lock = (
    ReadWriteLock(path)
    if kind == "ReadWriteLock"
    else SoftFileLock(path, timeout=0)
)
context = lock.read_lock(timeout=0) if kind == "ReadWriteLock" else lock
try:
    with context:
        print("acquired")
except filelock.Timeout:
    print("blocked")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), lock_kind],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip()


def test_security_patch_version_is_installed() -> None:
    """The runtime contains the selected patched filelock release."""
    assert filelock.__version__ == "4.2.4"


@pytest.mark.parametrize("lock_kind", ["SoftFileLock", "ReadWriteLock"])
def test_lock_blocks_other_process_then_releases(
    tmp_path: Path, lock_kind: str
) -> None:
    """A separate process waits until each lock type is released."""
    path = tmp_path / "shared.lock"
    if lock_kind == "ReadWriteLock":
        lock = ReadWriteLock(str(path))
        held = lock.write_lock(timeout=2)
    else:
        lock = SoftFileLock(path, timeout=2)
        held = lock
    with held:
        assert _child_attempt(path, lock_kind) == "blocked"
    assert _child_attempt(path, lock_kind) == "acquired"
