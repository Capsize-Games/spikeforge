"""Hash a saved image's configuration independently of Docker storage mode."""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from typing import IO

MAX_ENTRIES = 4096
MAX_MANIFEST_BYTES = 1024 * 1024


def file_digest(stream: IO[bytes]) -> str:
    """Hash actual archive bytes while retaining only a bounded buffer."""
    digest = hashlib.sha256()
    while block := stream.read(65536):
        digest.update(block)
    return "sha256:" + digest.hexdigest()


def record_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo, hashes: dict[str, str],
) -> bytes | None:
    """Read bounded manifest metadata or hash a file without extraction."""
    if not member.isfile():
        return None
    if len(member.name.encode()) > 1024 or member.name in hashes:
        raise ValueError("Invalid or duplicate saved image member")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError("Saved image member is missing")
    with source:
        if member.name == "manifest.json":
            if member.size > MAX_MANIFEST_BYTES:
                raise ValueError("Saved image manifest exceeds metadata limit")
            return source.read()
        hashes[member.name] = file_digest(source)
    return None


def selected_digest(manifest: bytes | None, hashes: dict[str, str]) -> str:
    """Require one manifest image and its actual configuration bytes."""
    if manifest is None:
        raise ValueError("Saved image manifest is missing")
    images: object = json.loads(manifest)
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("Expected exactly one saved image")
    image: object = images[0]
    if not isinstance(image, dict):
        raise ValueError("Saved image manifest entry is invalid")
    config: object = image.get("Config")
    if not isinstance(config, str) or config not in hashes:
        raise ValueError("Saved image configuration is missing")
    return hashes[config]


def configuration_digest(stream: IO[bytes]) -> str:
    """Process configs before or after the manifest with bounded memory."""
    hashes: dict[str, str] = {}
    manifest = None
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for index, member in enumerate(archive):
            if index >= MAX_ENTRIES:
                raise ValueError("Saved image exceeds metadata entry limit")
            metadata = record_member(archive, member, hashes)
            if metadata is not None:
                if manifest is not None:
                    raise ValueError("Duplicate saved image manifest")
                manifest = metadata
    return selected_digest(manifest, hashes)


if __name__ == "__main__":
    print(configuration_digest(sys.stdin.buffer))
