"""``GET /api/bundle/{name}``: download a saved checkpoint as a .spkf file.

Called directly as a plain coroutine via ``asyncio.run`` rather than through
an HTTP client -- ``httpx`` isn't a project dependency (see
``tests/test_serve_app.py``'s own note) and a FastAPI route handler is just
an ``async def`` underneath the decorator, so this covers the real code path
without adding one.
"""

import asyncio
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi import HTTPException

from server.bundle_routes import download_bundle
from spikeforge.network import model_store
from spikeforge.serving import bundle_manifest as bm
from spikeforge.training.training_engine import TrainingEngine

_NAME = "bundle_route_ckpt"


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint reads/writes into a per-test directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _save_checkpoint() -> None:
    """Train one real step and save a small fc_small checkpoint."""
    torch.manual_seed(0)
    engine = TrainingEngine(
        dataset="mnist",
        num_steps=4,
        device="cpu",
        topology="fc_small",
        topology_params={"hidden": 5, "num_classes": 3},
    )
    engine.save(_NAME)


def test_download_bundle_returns_a_valid_spkf_file() -> None:
    """A saved checkpoint downloads as a real, loadable .spkf archive."""
    _save_checkpoint()

    response = asyncio.run(download_bundle(_NAME))

    assert response.filename == f"{_NAME}.spkf"
    assert response.media_type == "application/octet-stream"
    path = Path(response.path)
    assert path.exists()
    with zipfile.ZipFile(path) as archive:
        assert bm.MANIFEST_NAME in archive.namelist()
        assert bm.WEIGHTS_NAME in archive.namelist()


def test_download_bundle_cleans_up_after_response() -> None:
    """The background task removes the temp directory once it has run."""
    _save_checkpoint()

    response = asyncio.run(download_bundle(_NAME))
    workdir = Path(response.path).parent
    assert workdir.exists()

    asyncio.run(response.background())

    assert not workdir.exists()


def test_download_bundle_404s_an_unknown_name() -> None:
    """A name with no saved checkpoint is a 404, not a 500 or a raw error."""
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(download_bundle("nope"))
    assert excinfo.value.status_code == 404
