"""Checksum and size verification honesty."""

import hashlib
from pathlib import Path

from spikeforge_hub import verify


def _written(tmp_path: Path, blob: bytes) -> str:
    """Write ``blob`` to an artifact file and return its path."""
    path = tmp_path / "artifact.bin"
    path.write_bytes(blob)
    return str(path)


def test_matching_checksum_verifies(tmp_path: Path) -> None:
    """A published checksum that matches yields ``verified``."""
    blob = b"spikes"
    path = _written(tmp_path, blob)
    digest = hashlib.sha256(blob).hexdigest()
    result = verify.verify_file(path, digest, len(blob))
    assert result.status == verify.VERIFIED
    assert result.verified is True
    assert result.sha256 == digest


def test_checksum_mismatch_fails(tmp_path: Path) -> None:
    """A seeded checksum mismatch is a hard, named failure."""
    path = _written(tmp_path, b"real")
    result = verify.verify_file(path, "0" * 64)
    assert result.status == verify.FAILED
    assert result.failed is True
    assert "sha256 mismatch" in (result.reason or "")


def test_size_mismatch_fails(tmp_path: Path) -> None:
    """A size mismatch is a hard, named failure."""
    path = _written(tmp_path, b"real")
    result = verify.verify_file(path, size_bytes=999)
    assert result.status == verify.FAILED
    assert "size mismatch" in (result.reason or "")


def test_absent_checksum_is_unverified(tmp_path: Path) -> None:
    """No published checksum reports unverified, never verified."""
    path = _written(tmp_path, b"real")
    result = verify.verify_file(path)
    assert result.status == verify.UNVERIFIED
    assert result.verified is False
    assert result.reason


def test_directory_size_match_stays_unverified(tmp_path: Path) -> None:
    """A directory snapshot has no checksum, so it stays unverified."""
    (tmp_path / "a.bin").write_bytes(b"12")
    result = verify.verify_dir(str(tmp_path), size_bytes=2)
    assert result.status == verify.UNVERIFIED
    assert result.sha256 is None


def test_directory_size_mismatch_fails(tmp_path: Path) -> None:
    """A mismatched directory size is a hard failure."""
    (tmp_path / "a.bin").write_bytes(b"12")
    result = verify.verify_dir(str(tmp_path), size_bytes=5)
    assert result.status == verify.FAILED
