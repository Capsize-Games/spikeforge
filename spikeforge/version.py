"""Report the installed versions of every spikeforge distribution.

The workspace publishes seven independently-versioned distributions, so
"what version am I running?" has seven answers and the interesting one is
whether the *combination* is a shipped release. ``compatibility.json`` is the
source of truth for that; before this module it was a file you had to know to
go and read, invisible to anyone filing a bug report.

Versions are read from installed distribution metadata rather than from a
hard-coded string, so ``__version__`` cannot drift from the wheel.
"""

import argparse
import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Every distribution this workspace publishes, core first. ``dashboard``
#: appears in ``compatibility.json`` but is not a Python distribution, so it
#: is not listed here.
DISTRIBUTIONS = (
    "spikeforge",
    "spikeforge-targets",
    "spikeforge-hub",
    "spikeforge-io",
    "spikeforge-server",
    "spikeforge-serve",
    "spikeforge-clients",
)

#: Reported when the package is imported from a source tree that was never
#: installed, so there is no metadata to read.
UNKNOWN = "0.0.0+unknown"

_MATRIX_URL = (
    "https://github.com/Capsize-Games/spikeforge/blob/main/compatibility.json"
)


def distribution_version(name: str) -> Optional[str]:
    """Return the installed version of ``name``, or None if it is absent."""
    try:
        return _metadata_version(name)
    except PackageNotFoundError:
        return None


def installed_versions() -> Dict[str, str]:
    """Return every spikeforge distribution present, in declaration order."""
    found = {}
    for name in DISTRIBUTIONS:
        installed = distribution_version(name)
        if installed is not None:
            found[name] = installed
    return found


def _matrix_path() -> Optional[Path]:
    """Locate ``compatibility.json``, packaged or in a source checkout."""
    packaged = Path(__file__).resolve().parent / "compatibility.json"
    if packaged.is_file():
        return packaged
    # An editable install points at the source tree, where the file is the
    # repository-root original the packaged copy symlinks to.
    checkout = Path(__file__).resolve().parents[1] / "compatibility.json"
    return checkout if checkout.is_file() else None


def compatibility_releases() -> List[Dict[str, str]]:
    """Return the released version combinations, newest last."""
    path = _matrix_path()
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    releases = payload.get("releases", [])
    return [release for release in releases if isinstance(release, dict)]


def _unrecorded_message(core: str, releases: List[Dict[str, str]]) -> str:
    """Return the message for a core version absent from the matrix."""
    known = sorted({r.get("spikeforge", "") for r in releases})
    return (
        f"compatibility: UNRECORDED -- spikeforge {core} is not in "
        f"compatibility.json (recorded: {', '.join(known)}).\n"
        f"  {_MATRIX_URL}"
    )


def _matches_release(
    candidates: List[Dict[str, str]], satellites: Dict[str, str]
) -> bool:
    """Return True when some candidate release matches every satellite."""
    return any(
        all(release.get(n) == v for n, v in satellites.items())
        for release in candidates
    )


def _mismatched_satellites(
    core: str, candidates: List[Dict[str, str]], satellites: Dict[str, str]
) -> List[str]:
    """Return one description per satellite version outside the matrix."""
    mismatched = []
    for name, found in sorted(satellites.items()):
        expected = sorted({str(r[name]) for r in candidates if r.get(name)})
        if found not in expected:
            mismatched.append(
                f"{name} {found} (recorded with spikeforge {core}: "
                f"{', '.join(expected) or 'none'})"
            )
    return mismatched


def _mismatch_message(
    core: str, candidates: List[Dict[str, str]], satellites: Dict[str, str]
) -> str:
    """Return the MISMATCH message naming every satellite outside range."""
    mismatched = _mismatched_satellites(core, candidates, satellites)
    detail = "; ".join(mismatched) or "an unrecorded combination"
    return (
        f"compatibility: MISMATCH -- {detail}.\n"
        f"  Pin from the matrix rather than assuming semver alignment: "
        f"{_MATRIX_URL}"
    )


def compatibility_status() -> str:
    """Describe whether the installed combination is a shipped release.

    Only distributions that are actually installed are considered: a core-only
    install is a supported combination, not a mismatch.
    """
    installed = installed_versions()
    core = installed.get("spikeforge")
    releases = compatibility_releases()
    if not releases:
        return "compatibility: unknown (compatibility.json not found)"
    if core is None:
        return "compatibility: unknown (core distribution not installed)"
    candidates = [r for r in releases if r.get("spikeforge") == core]
    if not candidates:
        return _unrecorded_message(core, releases)
    satellites = {n: v for n, v in installed.items() if n != "spikeforge"}
    if _matches_release(candidates, satellites):
        return "compatibility: OK (a recorded release combination)"
    return _mismatch_message(core, candidates, satellites)


def version_report() -> str:
    """Return the multi-line text printed by every ``--version`` flag."""
    installed = installed_versions()
    # Pad against every distribution, not only the installed ones: a
    # core-only install would otherwise print a ragged column.
    width = max(len(name) for name in DISTRIBUTIONS)
    lines = [f"spikeforge {installed.get('spikeforge', UNKNOWN)}"]
    for name in DISTRIBUTIONS[1:]:
        found = installed.get(name, "not installed")
        lines.append(f"  {name:<{width}}  {found}")
    lines.append(compatibility_status())
    return "\n".join(lines)


class _VersionAction(argparse.Action):
    """Print the version report verbatim, then exit 0.

    argparse's built-in ``version`` action pipes its string through the help
    formatter, which reflows it into a single wrapped paragraph. The report is
    a table, so it has to keep its own line breaks -- and it belongs on stdout,
    which ``parser.exit(message=...)`` would not do.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        """Write the report to stdout and stop argument parsing."""
        print(version_report())
        parser.exit()


def add_version_flag(parser: argparse.ArgumentParser) -> None:
    """Register ``--version`` on a CLI that already imports the package.

    The report is resolved only when the flag is used. The two tutorial entry
    points (``main.py``, ``main_encodings.py``) deliberately do not import
    this module at all: they keep ``--help`` free of the spikeforge package,
    and therefore of torch, which is what makes them answer in milliseconds.
    """
    parser.add_argument(
        "--version",
        action=_VersionAction,
        nargs=0,
        default=argparse.SUPPRESS,
        help="print the installed spikeforge versions and exit",
    )


#: The core distribution's version, or :data:`UNKNOWN` when the package is
#: imported from a source tree that was never installed.
__version__ = distribution_version("spikeforge") or UNKNOWN
