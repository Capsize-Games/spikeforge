"""A ``.spkf`` whose weights are compressed loads and predicts closely."""

from typing import Any

import pytest
import torch

from spikeforge.network import model_store
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle, build
from spikeforge.serving.session import InferenceSession
from spikeforge.training.training_engine import TrainingEngine

_NAME = "compressed_ckpt"
#: Largest logit difference the int8 round-trip may introduce.
TOLERANCE = 1e-2


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint reads and writes into a per-test directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _checkpoint() -> str:
    """Save a small fc_small checkpoint and return its name."""
    torch.manual_seed(0)
    engine = TrainingEngine(
        dataset="mnist",
        num_steps=4,
        device="cpu",
        topology="fc_small",
        topology_params={"hidden": 5, "num_classes": 3},
    )
    engine.save(_NAME)
    return _NAME


def _frame() -> torch.Tensor:
    """Return one fixed input frame for both bundles."""
    torch.manual_seed(7)
    return torch.rand(1, 28 * 28)


def test_compressed_bundle_loads_and_predicts_within_tolerance(
    tmp_path: Any,
) -> None:
    """An int8 bundle rebuilds and its session tracks the float reference."""
    reference = build(_checkpoint())
    compressed = build(_checkpoint(), compress="int8")
    frame = _frame()
    plain = InferenceSession.load(reference).step(frame)
    quantized = InferenceSession.load(compressed).step(frame)
    assert torch.allclose(
        plain.logits, quantized.logits, atol=TOLERANCE
    )
    assert plain.label == quantized.label


def test_manifest_records_a_versioned_weights_encoding(tmp_path: Any) -> None:
    """The manifest names the scheme, its version, and a ratio above one."""
    compressed = build(_checkpoint(), compress="int8")
    encoding = compressed.manifest["weights_encoding"]
    assert encoding["scheme"] == "int8"
    assert encoding["version"] == 1
    assert encoding["bits"] == 8
    assert encoding["ratio"] > 1.0
    report = compressed.compression_report()
    assert report is not None
    assert report.scheme == "int8"
    assert report.ratio > 1.0


def test_compressed_archive_is_smaller_but_self_consistent(
    tmp_path: Any,
) -> None:
    """The written archive shrinks while every required entry survives."""
    raw = build(_checkpoint()).to_bytes()
    out = str(tmp_path / "model.spkf")
    compressed = build(_checkpoint(), out=out, compress="uint8")
    assert len(compressed.to_bytes()) < len(raw)
    loaded = DeploymentBundle.load(out)
    assert loaded.weights_encoding is not None
    assert loaded.weights_encoding["scheme"] == "uint8"
    with open(out, "rb") as handle:
        assert handle.read(2) == b"PK"


def test_loaded_module_matches_the_dequantized_weights(
    tmp_path: Any,
) -> None:
    """``build_module`` loads the dequantized payload, not the raw codes."""
    out = str(tmp_path / "model.spkf")
    build(_checkpoint(), out=out, compress="int8")
    loaded = DeploymentBundle.load(out)
    restored = loaded.resolved_weights()
    encoded = set((loaded.weights_encoding or {}).get("tensors") or {})
    assert encoded
    for key in encoded:
        assert restored[key].is_floating_point()
    rebuilt = loaded.build_module().state_dict()
    for key, value in restored.items():
        assert torch.equal(rebuilt[key], value)
    raw = build(_checkpoint()).resolved_weights()
    for key, value in raw.items():
        assert torch.allclose(
            rebuilt[key], value, atol=TOLERANCE
        )


def test_pruned_and_compressed_bundle_reports_both(tmp_path: Any) -> None:
    """Pruning folds into the manifest and the compression report."""
    bundle = build(
        _checkpoint(),
        compress="int8",
        prune_sparsity=0.5,
        prune_strategy="unstructured",
    )
    pruning = bundle.manifest["pruning"]
    assert pruning["sparsity"] > 0.0
    assert pruning["drift"]["max_abs"] >= 0.0
    report = bundle.compression_report()
    assert report is not None
    assert report.pruning is not None
    assert report.pruning["strategy"] == "unstructured"
    bundle.build_module()


def test_raw_bundle_reports_no_compression() -> None:
    """A bundle built without a scheme has no encoding and no report."""
    bundle = build(_checkpoint())
    assert bundle.weights_encoding is None
    assert bundle.compression_report() is None
    assert bundle.manifest["weights_encoding"] is None
    assert bundle.manifest["pruning"] is None


def test_bundle_entry_set_is_unchanged() -> None:
    """Compression reuses ``weights.pt`` rather than adding an entry."""
    import zipfile

    compressed = build(_checkpoint(), compress="int8")
    payload = compressed.to_bytes()
    import io

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(bm.REQUIRED_ENTRIES) <= set(archive.namelist())
        assert set(bm.OPTIONAL_ENTRIES).isdisjoint(
            set(archive.namelist())
        )
