"""Checksum and size verification for downloaded hub artifacts.

Verification is honest by construction: a published checksum that does not
match is a hard failure, while a source that publishes no checksum is reported
as *unverified* rather than passing silently. Directory snapshots (Hugging
Face repos) have no single checksum, so only their summed size can be checked.
"""

import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from spikeforge_hub import cache

#: Verification status values, matching the honest report vocabulary.
VERIFIED = "verified"
UNVERIFIED = "unverified"
FAILED = "failed"

#: Read size used while hashing a file, so large artifacts stream.
_CHUNK = 1 << 20


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of verifying one artifact against its declared metadata."""

    status: str
    reason: Optional[str]
    sha256: Optional[str]
    size_bytes: int

    @property
    def verified(self) -> bool:
        """Return True when the artifact matched its published checksum."""
        return self.status == VERIFIED

    @property
    def failed(self) -> bool:
        """Return True when a published expectation was not met."""
        return self.status == FAILED


def file_sha256(path: str, chunk: int = _CHUNK) -> str:
    """Return the hex sha256 digest of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(
    path: str,
    sha256: Optional[str] = None,
    size_bytes: Optional[int] = None,
) -> VerificationResult:
    """Verify one file against an optional checksum and size."""
    actual_sha = file_sha256(path)
    actual_size = os.path.getsize(path)
    return _verdict(actual_sha, actual_size, sha256, size_bytes)


def verify_dir(
    path: str,
    size_bytes: Optional[int] = None,
) -> VerificationResult:
    """Verify a directory snapshot by summed size (checksum unavailable)."""
    total = cache.dir_bytes(path)
    if size_bytes is not None and total != size_bytes:
        reason = f"size mismatch: expected {size_bytes}, got {total}"
        return VerificationResult(FAILED, reason, None, total)
    return VerificationResult(
        UNVERIFIED,
        "directory snapshot; no per-file checksum published",
        None,
        total,
    )


def _verdict(
    actual_sha: str,
    actual_size: int,
    sha256: Optional[str],
    size_bytes: Optional[int],
) -> VerificationResult:
    """Return the honest verdict for measured values and expectations."""
    status, reason = _outcome(actual_sha, actual_size, sha256, size_bytes)
    return VerificationResult(status, reason, actual_sha, actual_size)


def _outcome(
    actual_sha: str,
    actual_size: int,
    sha256: Optional[str],
    size_bytes: Optional[int],
) -> Tuple[str, Optional[str]]:
    """Return the status and reason for measured values and expectations."""
    if sha256 is not None and actual_sha != sha256:
        return FAILED, f"sha256 mismatch: expected {sha256}, got {actual_sha}"
    if size_bytes is not None and actual_size != size_bytes:
        return (
            FAILED,
            f"size mismatch: expected {size_bytes}, got {actual_size}",
        )
    if sha256 is None:
        return UNVERIFIED, "no checksum published for this artifact"
    return VERIFIED, None
