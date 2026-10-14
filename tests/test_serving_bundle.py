"""The portable deployment bundle: build, verify, and load."""

import os
import pathlib
import subprocess
import sys
import zipfile
from typing import Any

import pytest
import torch
from _bundle_helpers import _NAME, _checkpoint, _written

from spikeforge.network import model_store
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle, build


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint reads and writes into a per-test directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def test_round_trip_rebuilds_module_and_weights(tmp_path: Any) -> None:
    """A written bundle reloads into the same weights and module."""
    out = str(tmp_path / "model.spkf")
    built = build(_checkpoint(), out=out)
    loaded = DeploymentBundle.load(out)
    assert set(loaded.weights) == set(built.weights)
    for key, value in built.weights.items():
        assert torch.equal(value, loaded.weights[key])
    assert set(loaded.build_module().state_dict()) == set(loaded.weights)


def test_manifest_records_spec_versions_and_entries(tmp_path: Any) -> None:
    """The manifest is self-describing and the entry set is complete."""
    loaded = DeploymentBundle.load(_written(tmp_path))
    manifest = loaded.manifest
    assert manifest["topology"] == "fc_small"
    stages = {s["name"]: s for s in manifest["spec"]["stages"]}
    assert stages["fc2"]["params"]["out_features"] == 3
    assert manifest["spec"]["input"] == "flatten"
    assert manifest["library_versions"]["torch"]
    assert manifest["encode_spec_version"] == bm.ENCODE_SPEC_VERSION
    assert loaded.encode_config["spec_version"] == bm.ENCODE_SPEC_VERSION
    assert loaded.encode_config["coding"] == "rate"
    assert loaded.has_encode() is True
    assert loaded.preprocessing == {}
    with zipfile.ZipFile(loaded.path or "") as archive:
        assert set(bm.REQUIRED_ENTRIES) <= set(archive.namelist())


def test_rebuild_is_bit_exact_in_a_subprocess(tmp_path: Any) -> None:
    """A fresh process rebuilds the module bit-for-bit from the bundle."""
    out = _written(tmp_path)
    built = build(_NAME)
    reference = tmp_path / "reference.pt"
    torch.save(dict(built.weights), reference)
    lines = [
        "import sys, torch",
        "from spikeforge.serving import DeploymentBundle",
        "bundle = DeploymentBundle.load(sys.argv[1])",
        "module = bundle.build_module()",
        "wanted = torch.load(sys.argv[2], map_location='cpu',",
        "                    weights_only=True)",
        "for key, value in module.state_dict().items():",
        "    assert torch.equal(value, wanted[key]), key",
        "print('ok')",
    ]
    root = str(pathlib.Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", "\n".join(lines), out, str(reference)],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=root),
        capture_output=True,
        text=True,
        # Generous because this is a cold Python process importing Torch, and
        # the runners share one workstation with every other project's CI. At
        # 180s it timed out under load and reported a bit-exactness failure
        # that had not happened, which is the worst kind of red: it looks like
        # the thing the test is named after.
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
