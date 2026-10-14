"""Torch-free CLI checks for ``spikeforge-encodings`` (issue #117).

``main_encodings`` keeps ``--help`` (and ``--version``) free of the ML
stack by parsing arguments before importing anything heavy. These tests
run the entry point in a subprocess with the ML stack made
unimportable, so a future eager import fails here instead of shipping a
CLI that needs torch to print its own usage.
"""

import subprocess
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent.parent

#: Child preamble that makes the ML stack unimportable, mirroring a
#: genuinely absent install (``sys.modules[name] = None``) rather than a
#: partially initialised one.
_BLOCK_ML_STACK = (
    "import sys\n"
    "for _m in ('torch', 'snntorch', 'torchvision', 'matplotlib'):\n"
    "    sys.modules[_m] = None\n"
    "assert sys.modules['torch'] is None\n"
)


def _run_help_or_version(args: List[str]) -> subprocess.CompletedProcess[str]:
    """Run ``main_encodings.main`` with ``args`` and no ML stack."""
    script = (
        _BLOCK_ML_STACK + "import main_encodings\n"
        f"main_encodings.main({args!r})\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_help_is_torch_free() -> None:
    """``--help`` prints usage with the ML stack unimportable."""
    result = _run_help_or_version(["--help"])
    assert result.returncode == 0, result.stderr
    assert "spikeforge-encodings" in result.stdout
    assert "torch" not in result.stderr


def test_version_is_torch_free() -> None:
    """``--version`` prints the report with the ML stack unimportable."""
    result = _run_help_or_version(["--version"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("spikeforge ")
    assert "compatibility:" in result.stdout
