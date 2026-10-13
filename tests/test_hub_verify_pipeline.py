"""End-to-end: load, funnel, and energy-account a real bundle."""

import zipfile
from typing import Any, Dict

import pytest
import torch
from hub_verify.energy import energy_reports
from hub_verify.errors import VerificationStepError
from hub_verify.funnel import run_funnel
from hub_verify.pipeline import load_and_build

from spikeforge.network import model_store
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import build
from spikeforge.training.training_engine import TrainingEngine

_NAME = "hub_verify_ckpt"


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint reads and writes into a per-test directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _written(tmp_path: Any) -> str:
    """Save a small checkpoint, build its bundle, and return its path."""
    torch.manual_seed(0)
    engine = TrainingEngine(
        dataset="mnist",
        num_steps=4,
        device="cpu",
        topology="fc_small",
        topology_params={"hidden": 5, "num_classes": 3},
    )
    engine.save(_NAME)
    out = str(tmp_path / "model.spkf")
    build(_NAME, out=out)
    return out


def _tamper_weights(path: str) -> None:
    """Flip a byte of the stored weights, invalidating its checksum."""
    with zipfile.ZipFile(path) as archive:
        entries = {n: archive.read(n) for n in archive.namelist()}
    entries[bm.WEIGHTS_NAME] += b"\x00"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def test_a_good_bundle_passes_the_whole_funnel(tmp_path: Any) -> None:
    """A real, untampered bundle loads, funnels, and accounts cleanly."""
    out = _written(tmp_path)
    bundle, module = load_and_build(out)
    funnel: Dict[str, Any] = run_funnel(bundle, module, str(tmp_path))
    assert funnel["inspect"]["kind"] == "state_dict"
    assert funnel["drift"]["within_tolerance"] is True
    reports = energy_reports(bundle.spec, module)
    assert reports
    assert all("target" in report for report in reports)


def test_a_tampered_bundle_is_rejected_at_the_bundle_stage(
    tmp_path: Any,
) -> None:
    """A checksum mismatch names the bundle stage, not a generic failure."""
    out = _written(tmp_path)
    _tamper_weights(out)
    with pytest.raises(VerificationStepError) as info:
        load_and_build(out)
    assert info.value.step == "bundle"
    assert info.value.reason
