"""Built distributions must retain the target energy-accounting tables."""

import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Mapping

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE = _ROOT / "packages" / "spikeforge-targets"
_TABLES = _ROOT / "spikeforge_targets" / "energy" / "costs"
_EXPECTED = {path.stem for path in _TABLES.glob("*.json")}
_PROBE = (
    "import json, sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from spikeforge_targets.energy import account, target_costs\n"
    "print(json.dumps({'names': target_costs.names(), "
    "'basis': account({'sop': 1, 'mac': 2, 'ac': 1}, 'reference').basis}))\n"
)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, Path]:
    """Build wheel and sdist from the target distribution metadata."""
    output = tmp_path_factory.mktemp("package-build")
    result = subprocess.run(
        ["uv", "build", "--out-dir", str(output)],
        cwd=_PACKAGE, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return {
        "wheel": next(output.glob("*.whl")),
        "sdist": next(output.glob("*.tar.gz")),
    }


def _extract(artifact: Path, destination: Path) -> Path:
    """Extract the locally built package to an isolated runtime probe path."""
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            archive.extractall(destination)
        return destination
    with tarfile.open(artifact) as archive:
        archive.extractall(destination, filter="data")
    return next(destination.iterdir())


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_distribution_contains_energy_costs(
    artifacts: Mapping[str, Path], tmp_path: Path, kind: str,
) -> None:
    """Installed wheel and source artifacts retain every declared cost."""
    root = _extract(artifacts[kind], tmp_path)
    packaged = root / "spikeforge_targets" / "energy" / "costs"
    assert {path.stem for path in packaged.glob("*.json")} == _EXPECTED
    for name in _EXPECTED:
        assert (packaged / f"{name}.json").read_bytes() == (
            _TABLES / f"{name}.json"
        ).read_bytes()


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_distribution_runtime_reports_declared_energy(
    artifacts: Mapping[str, Path], tmp_path: Path, kind: str,
) -> None:
    """An isolated artifact import yields estimates instead of unavailable."""
    root = _extract(artifacts[kind], tmp_path)
    result = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE, str(root)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    assert set(payload["names"]) == _EXPECTED
    assert payload["basis"] == "declared cost table"
