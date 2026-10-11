"""Validate bounded streaming identity from actual Docker archive format."""

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = b'{"architecture":"amd64","os":"linux","config":{}}'


def archive(entries: list[tuple[str, bytes]]) -> bytes:
    """Serialize archive input; this is data, not a replacement Docker API."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as output:
        for name, content in entries:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))
    return stream.getvalue()


def digest(
    entries: list[tuple[str, bytes]],
) -> subprocess.CompletedProcess[bytes]:
    """Feed the actual command a byte stream, retaining failure details."""
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/dashboard-image-digest.py")],
        input=archive(entries), capture_output=True,
    )


def test_config_before_manifest_with_no_json_suffix_is_hashed() -> None:
    """Containerd config names and archive order preserve actual byte ID."""
    path = "blobs/sha256/arbitrary-name"
    manifest = json.dumps([{"Config": path, "RepoTags": None}]).encode()
    result = digest([(path, CONFIG), ("manifest.json", manifest)])
    expected = "sha256:" + hashlib.sha256(CONFIG).hexdigest()
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().strip() == expected


def test_multiple_saved_images_are_rejected() -> None:
    """An ambiguous archive cannot identify a single expected image."""
    manifest = json.dumps([{"Config": "a"}, {"Config": "b"}]).encode()
    result = digest([("manifest.json", manifest)])
    assert result.returncode != 0
    assert b"Expected exactly one saved image" in result.stderr


def test_missing_config_is_rejected() -> None:
    """The manifest must reference bytes actually present in the archive."""
    manifest = json.dumps([{"Config": "absent"}]).encode()
    result = digest([("manifest.json", manifest)])
    assert result.returncode != 0
    assert b"Saved image configuration is missing" in result.stderr


def test_oversized_manifest_is_rejected() -> None:
    """Archive metadata cannot request an unbounded allocation."""
    result = digest([("manifest.json", b" " * (1024 * 1024 + 1))])
    assert result.returncode != 0
    assert b"Saved image manifest exceeds metadata limit" in result.stderr
