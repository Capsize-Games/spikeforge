"""Version reporting across the seven distributions and the CLI flag."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from spikeforge import version as version_module
from spikeforge.version import (
    DISTRIBUTIONS,
    UNKNOWN,
    add_version_flag,
    compatibility_releases,
    compatibility_status,
    distribution_version,
    installed_versions,
    version_report,
)

_ROOT = Path(__file__).resolve().parent.parent
#: Every import root that must expose ``__version__``, with the distribution
#: whose metadata it reads. ``spikeforge_clients`` is deliberately excluded
#: from the import test: it is not installed in every CI job.
_ROOTS = {
    "spikeforge": "spikeforge",
    "spikeforge_targets": "spikeforge-targets",
    "spikeforge_hub": "spikeforge-hub",
    "spikeforge_io": "spikeforge-io",
    "server": "spikeforge-server",
}


def _release(**overrides: str) -> Dict[str, str]:
    """Return a compatibility-matrix row with sensible defaults."""
    row = {
        "spikeforge": "0.3.4",
        "spikeforge-targets": "0.1.1",
        "spikeforge-hub": "0.1.0",
    }
    row.update(overrides)
    return row


def test_core_exposes_version() -> None:
    """``import spikeforge`` answers the first question in a bug report."""
    import spikeforge

    assert spikeforge.__version__ == distribution_version("spikeforge")
    assert spikeforge.__version__ != UNKNOWN
    assert "__version__" in spikeforge.__all__


@pytest.mark.parametrize(("root", "dist"), sorted(_ROOTS.items()))
def test_every_import_root_exposes_its_own_version(
    root: str, dist: str
) -> None:
    """Each distribution reports its own version, not core's.

    An import root is importable from a source checkout whether or not its
    distribution was installed -- the repository root is on ``sys.path`` -- and
    several CI jobs deliberately install only part of the workspace. Both
    states are contractual: installed reports the metadata version, absent
    reports the explicit unknown sentinel. Never core's version, and never a
    guess.
    """
    module = pytest.importorskip(root)
    installed = distribution_version(dist)
    if installed is None:
        assert module.__version__ == UNKNOWN
    else:
        assert module.__version__ == installed


def test_installed_versions_lists_only_what_is_present() -> None:
    """A distribution that is not installed is omitted, never guessed."""
    found = installed_versions()
    assert "spikeforge" in found
    assert set(found) <= set(DISTRIBUTIONS)
    for name, reported in found.items():
        assert reported == distribution_version(name)


def test_compatibility_matrix_is_readable_from_an_installed_package() -> None:
    """The matrix ships with the wheel, so ``--version`` can read it."""
    releases = compatibility_releases()
    assert releases, "compatibility.json was not found from the package"
    matrix = json.loads(
        (_ROOT / "compatibility.json").read_text(encoding="utf-8")
    )
    assert releases == matrix["releases"]


def test_packaged_matrix_is_the_repository_matrix() -> None:
    """The packaged copy is a symlink, so it cannot drift from the root."""
    packaged = _ROOT / "spikeforge" / "compatibility.json"
    assert packaged.is_file()
    assert packaged.resolve() == (_ROOT / "compatibility.json").resolve()


def test_status_reports_ok_for_a_recorded_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed set that matches a matrix row is reported as OK."""
    monkeypatch.setattr(
        version_module,
        "installed_versions",
        lambda: {"spikeforge": "0.3.4", "spikeforge-targets": "0.1.1"},
    )
    monkeypatch.setattr(
        version_module, "compatibility_releases", lambda: [_release()]
    )
    assert compatibility_status().startswith("compatibility: OK")


def test_status_flags_a_satellite_off_the_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mismatch names the satellite, what is installed, and what fits."""
    monkeypatch.setattr(
        version_module,
        "installed_versions",
        lambda: {"spikeforge": "0.3.4", "spikeforge-targets": "0.2.0"},
    )
    monkeypatch.setattr(
        version_module, "compatibility_releases", lambda: [_release()]
    )
    status = compatibility_status()
    assert status.startswith("compatibility: MISMATCH")
    assert "spikeforge-targets 0.2.0" in status
    assert "0.1.1" in status


def test_status_flags_an_unrecorded_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A core release the matrix has never seen is called out as such."""
    monkeypatch.setattr(
        version_module,
        "installed_versions",
        lambda: {"spikeforge": "9.9.9"},
    )
    monkeypatch.setattr(
        version_module, "compatibility_releases", lambda: [_release()]
    )
    status = compatibility_status()
    assert status.startswith("compatibility: UNRECORDED")
    assert "9.9.9" in status


def test_core_only_install_is_not_a_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installing core alone is a supported combination, not an error."""
    monkeypatch.setattr(
        version_module, "installed_versions", lambda: {"spikeforge": "0.3.4"}
    )
    monkeypatch.setattr(
        version_module, "compatibility_releases", lambda: [_release()]
    )
    assert compatibility_status().startswith("compatibility: OK")


def test_report_names_every_distribution() -> None:
    """The report is the whole picture, including what is absent."""
    report = version_report()
    for name in DISTRIBUTIONS:
        assert name in report
    assert report.startswith("spikeforge ")
    assert "compatibility:" in report


def test_version_flag_keeps_the_report_unwrapped() -> None:
    """The built-in version action reflows the table; ours must not."""
    import argparse

    parser = argparse.ArgumentParser(prog="probe")
    add_version_flag(parser)
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--version"])
    assert exit_info.value.code == 0


@pytest.mark.parametrize(
    "command",
    [
        ["-m", "spikeforge.cli.verify"],
        ["-m", "spikeforge.cli.records_cli"],
        ["-m", "spikeforge.benchmark.cli"],
        ["-c", "import main; main.main(['--version'])"],
        ["-c", "import main_encodings; main_encodings.main(['--version'])"],
    ],
)
def test_every_console_script_answers_version(command: List[str]) -> None:
    """``--version`` exits 0 on stdout for all five console scripts."""
    argv: List[Any] = [sys.executable, *command]
    if command[0] == "-m":
        argv.append("--version")
    result = subprocess.run(
        argv, cwd=_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("spikeforge ")
    assert "compatibility:" in result.stdout
    # A multi-line table, not one reflowed paragraph.
    assert len(result.stdout.strip().splitlines()) >= len(DISTRIBUTIONS)
