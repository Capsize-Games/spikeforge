"""Push orchestration: ids, the two-phase upload, and push provenance."""

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
from _transfer_helpers import _save_checkpoint

from spikeforge.network import model_store
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge_hub import cache, transfer
from spikeforge_hub.errors import HubArtifactError

pytest.importorskip("nir")

@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the hub cache and model store into a temporary directory."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path / "hub"))
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path / "models"))
    return tmp_path


class _FakeClient:
    """A stand-in hub API recording the two-phase upload."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.intents: List[Dict[str, Any]] = []
        self.put: List[bytes] = []
        self.committed: List[str] = []

    def create_upload(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Record the intent and reserve a canned upload id."""
        self.intents.append(dict(intent))
        return {"upload_id": "u1"}

    def put_content(self, upload_id: str, data: bytes) -> Dict[str, Any]:
        """Record the bytes sent for ``upload_id``."""
        self.put.append(data)
        return {"upload_id": upload_id, "state": "received"}

    def commit_upload(self, upload_id: str) -> Dict[str, Any]:
        """Record the commit of ``upload_id`` as published."""
        self.committed.append(upload_id)
        return {"upload_id": upload_id, "state": "published"}


def _manifest_metrics(blob: bytes, dest: Path) -> Dict[str, Any]:
    """Return the ``expected_metrics`` recorded in ``.spkf`` bytes."""
    path = dest / "probe.spkf"
    path.write_bytes(blob)
    loaded = DeploymentBundle.load(str(path), strict=True)
    metrics = loaded.manifest.get("expected_metrics")
    assert isinstance(metrics, dict)
    return dict(metrics)


def test_push_runs_the_two_phase_upload(sandbox: Path) -> None:
    """Push builds the checkpoint and reserves, sends, and commits."""
    _save_checkpoint()
    fake = _FakeClient()
    report = transfer.push_saved_model(
        fake, "transfer_ckpt", version="1", license_id="BSD-3-Clause",
        slug="other-name",
    )
    intent = fake.intents[0]
    assert (intent["name"], intent["version"], intent["license"]) == (
        "other-name", "1", "BSD-3-Clause")
    assert intent["size_bytes"] == len(fake.put[0])
    names = set(zipfile.ZipFile(io.BytesIO(fake.put[0])).namelist())
    assert {bm.MANIFEST_NAME, bm.WEIGHTS_NAME} <= names
    assert fake.committed == ["u1"]
    assert (report["upload_id"], report["state"]) == ("u1", "published")
    assert report["sha256"] == intent["sha256"]


def test_push_unknown_model_names_it(sandbox: Path) -> None:
    """Pushing an unsaved name fails before any byte is built."""
    with pytest.raises(HubArtifactError, match="no saved model"):
        transfer.push_saved_model(
            _FakeClient(), "nope", version="1", license_id="BSD-3-Clause"
        )


def _manifest_metrics(blob: bytes, dest: Path) -> Dict[str, Any]:
    """Return the ``expected_metrics`` recorded in ``.spkf`` bytes."""
    path = dest / "probe.spkf"
    path.write_bytes(blob)
    loaded = DeploymentBundle.load(str(path), strict=True)
    metrics = loaded.manifest.get("expected_metrics")
    assert isinstance(metrics, dict)
    return dict(metrics)


def test_push_reference_id_builds_from_packaged_weights(
    sandbox: Path,
) -> None:
    """A reference id pushes its shipped checkpoint with its provenance.

    The licence, dataset declarations, and measured scores come from the
    catalog row, so staff do not restate (or mistype) them per push.
    """
    fake = _FakeClient()
    report = transfer.push_saved_model(
        fake, "reference/mnist-fc-legacy", version="1"
    )
    intent = fake.intents[0]
    assert intent["name"] == "mnist-fc-legacy"
    assert intent["license"] == "BSD-3-Clause"
    assert intent["dataset"] == "mnist"
    assert "MNIST" in intent["dataset_attribution"]
    expected = {
        "test_accuracy": 94.29,
        "test_samples": 10000,
        "test_dataset": "mnist",
    }
    assert _manifest_metrics(fake.put[0], sandbox) == expected
    assert report["expected_metrics"] == expected
    staged = [
        model["name"] for model in model_store.list_models()
        if model["name"].startswith("push-staging-")
    ]
    assert staged == []


def test_push_checkpoint_file_builds_outside_bytes(
    sandbox: Path,
) -> None:
    """``checkpoint`` pushes a .pt file's bytes under the given name."""
    _save_checkpoint()
    fake = _FakeClient()
    report = transfer.push_saved_model(
        fake,
        "from-file",
        version="1",
        license_id="BSD-3-Clause",
        checkpoint=model_store.path_for("transfer_ckpt"),
    )
    assert fake.intents[0]["name"] == "from-file"
    assert report["state"] == "published"
    assert _manifest_metrics(fake.put[0], sandbox) == {}


def test_push_missing_checkpoint_file_is_refused(sandbox: Path) -> None:
    """A checkpoint path that names no file is refused, not built."""
    with pytest.raises(HubArtifactError, match="no checkpoint file"):
        transfer.push_saved_model(
            _FakeClient(),
            "from-file",
            version="1",
            license_id="BSD-3-Clause",
            checkpoint=str(sandbox / "missing.pt"),
        )


def test_push_metrics_flags_land_in_the_manifest(
    sandbox: Path,
) -> None:
    """Claim flags are recorded into the bundle and echoed in the report."""
    _save_checkpoint()
    fake = _FakeClient()
    report = transfer.push_saved_model(
        fake,
        "transfer_ckpt",
        version="1",
        license_id="BSD-3-Clause",
        test_accuracy=94.29,
        test_samples=10000,
        test_dataset="mnist",
    )
    expected = {
        "test_accuracy": 94.29,
        "test_samples": 10000,
        "test_dataset": "mnist",
    }
    assert _manifest_metrics(fake.put[0], sandbox) == expected
    assert report["expected_metrics"] == expected


def test_push_without_metrics_records_empty(sandbox: Path) -> None:
    """Unstated scores ship ``{}``, never a zero that reads as measured."""
    _save_checkpoint()
    fake = _FakeClient()
    report = transfer.push_saved_model(
        fake, "transfer_ckpt", version="1", license_id="BSD-3-Clause"
    )
    assert _manifest_metrics(fake.put[0], sandbox) == {}
    assert report["expected_metrics"] == {}


def test_push_impossible_accuracy_is_refused(sandbox: Path) -> None:
    """A claim outside 0-100, or a non-number, fails like the catalog."""
    _save_checkpoint()
    for bad in (101.0, -1.0, "high"):
        with pytest.raises(HubArtifactError, match="test_accuracy"):
            transfer.push_saved_model(
                _FakeClient(),
                "transfer_ckpt",
                version="1",
                license_id="BSD-3-Clause",
                test_accuracy=bad,
            )


def test_push_samples_need_an_accuracy(sandbox: Path) -> None:
    """Qualifiers without a claim are refused rather than half-recorded."""
    _save_checkpoint()
    with pytest.raises(HubArtifactError, match="--test-accuracy"):
        transfer.push_saved_model(
            _FakeClient(),
            "transfer_ckpt",
            version="1",
            license_id="BSD-3-Clause",
            test_samples=10000,
        )


def test_push_description_reaches_the_intent(sandbox: Path) -> None:
    """A model-card body rides the intent the server persists."""
    _save_checkpoint()
    fake = _FakeClient()
    transfer.push_saved_model(
        fake,
        "transfer_ckpt",
        version="1",
        license_id="BSD-3-Clause",
        description="Intended for digit demos; not production OCR.",
    )
    intent = fake.intents[0]
    assert intent["description"] == (
        "Intended for digit demos; not production OCR."
    )
