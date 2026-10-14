"""Run git/gh and parse tags, stamps, and changed paths."""

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

#: Repository root (this file lives in ``scripts/``).
REPO_ROOT = Path(__file__).resolve().parent.parent


#: Release tags are ``<distribution>-v<version>``.
TAG_RE = re.compile(r"^(?P<dist>.+)-v(?P<version>\d+\.\d+.*)$")


def _run(command: Sequence[str]) -> Optional[str]:
    """Run ``command`` in the repo and return stdout, or None on failure."""
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _git(*args: str) -> Optional[str]:
    """Run ``git`` with ``args`` and return stdout, or None."""
    return _run(("git", *args))


def _gh_json(args: Sequence[str]) -> Optional[object]:
    """Return parsed ``gh`` JSON output, or None when it is unavailable."""
    output = _run(("gh", *args))
    if output is None:
        return None
    try:
        return json.loads(output)
    except ValueError:
        return None


def _parse_tag(tag: str) -> Optional[Tuple[str, str]]:
    """Split ``<distribution>-v<version>`` into its two parts."""
    match = TAG_RE.match(tag)
    if match is None:
        return None
    return match.group("dist"), match.group("version")


def _parse_stamp(stamp: str) -> Optional[datetime]:
    """Parse an ISO-8601 git timestamp into an aware datetime."""
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _tags() -> List[Tuple[str, datetime]]:
    """Return ``(tag, committed date)`` for every release tag, newest first."""
    output = _git(
        "for-each-ref",
        "--sort=-creatordate",
        "--format=%(refname:short) %(creatordate:iso-strict)",
        "refs/tags",
    )
    if not output:
        return []
    tags: List[Tuple[str, datetime]] = []
    for line in output.splitlines():
        tag, _, stamp = line.partition(" ")
        if _parse_tag(tag) is None or not stamp.strip():
            continue
        date = _parse_stamp(stamp.strip())
        if date is not None:
            tags.append((tag, date))
    return tags


def _changed_paths(rev_range: str) -> Tuple[str, ...]:
    """Return the files changed by ``rev_range`` (empty when unknown)."""
    output = _git("diff", "--name-only", rev_range)
    if not output:
        return ()
    return tuple(line for line in output.splitlines() if line.strip())

