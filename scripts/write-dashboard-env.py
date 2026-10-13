"""Persist secrets and the forced revision image in a private environment."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

KEY = "SPIKEFORGE_DASHBOARD_IMAGE"
IMAGE_ASSIGNMENT = re.compile(rf"^\s*(?:export\s+)?{KEY}\s*=")


def environment(content: str, revision: str) -> str:
    """Keep supplied secret lines, replacing every existing image override."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid deployment revision")
    lines = [line for line in content.splitlines()
             if not IMAGE_ASSIGNMENT.match(line)]
    lines.append(f"{KEY}=spikeforge-dashboard:{revision}")
    return "\n".join(lines) + "\n"


def write_environment(path: Path, content: str) -> None:
    """Atomically install mode 0600 without exposing secret file contents."""
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".env-")
    pending = Path(temporary)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


if __name__ == "__main__":
    target, source_revision = sys.argv[1:]
    write_environment(Path(target), environment(sys.stdin.read(),
                                               source_revision))
