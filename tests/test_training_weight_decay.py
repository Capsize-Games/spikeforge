"""Weight decay plumbing through the optimizer and checkpoint metadata."""

from typing import Any, Dict

import pytest

from spikeforge.network import model_store
from spikeforge.training.training_engine import TrainingEngine


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint writes into a per-test temporary directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _engine(**kwargs: Any) -> TrainingEngine:
    """Return a small download-free CPU engine."""
    defaults: Dict[str, Any] = {
        "dataset": "mnist", "num_steps": 4, "device": "cpu",
    }
    defaults.update(kwargs)
    return TrainingEngine(**defaults)


def test_default_weight_decay_is_zero() -> None:
    """An engine built without weight_decay keeps Adam's plain behaviour."""
    engine = _engine()
    assert engine._optimizer.defaults["weight_decay"] == 0.0


def test_weight_decay_reaches_the_optimizer() -> None:
    """A non-zero weight_decay is forwarded straight into Adam."""
    engine = _engine(weight_decay=1e-4)
    assert engine._optimizer.defaults["weight_decay"] == pytest.approx(1e-4)


def test_weight_decay_recorded_in_checkpoint_meta() -> None:
    """The saved checkpoint's meta records the weight_decay used to train."""
    engine = _engine(weight_decay=5e-3)
    engine.save("decayed")
    meta = model_store.load("decayed")["meta"]
    assert meta["weight_decay"] == pytest.approx(5e-3)


def test_weight_decay_recorded_in_manifest_hyperparameters() -> None:
    """The reproducibility manifest's config carries weight_decay too."""
    engine = _engine(weight_decay=5e-3, seed=1)
    engine.save("decayed_manifest")
    manifest = model_store.manifest("decayed_manifest")
    assert manifest["config"]["weight_decay"] == pytest.approx(5e-3)
