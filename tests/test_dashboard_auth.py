"""``server.auth``: the bearer/query-param gate for ``/ws`` and bundles.

Also covers the bundle route's own wiring: a bad auth-module integration
would still 401 the wrong requests even if ``authorized()`` itself were
correct in isolation.
"""

import asyncio
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch
from fastapi import HTTPException

from server.auth import DASHBOARD_TOKEN_ENV, authorized, configured_token
from server.bundle_routes import download_bundle
from spikeforge.network import model_store
from spikeforge.serving import bundle_manifest as bm
from spikeforge.training.training_engine import TrainingEngine

_NAME = "auth_route_ckpt"


class _FakeRequest:
    """Duck-types the one attribute ``download_bundle`` reads off Request."""

    def __init__(self, headers: Optional[Dict[str, str]] = None) -> None:
        """Store a plain header dict standing in for Starlette's."""
        self.headers: Dict[str, str] = headers or {}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the token env var starts unset for every test."""
    monkeypatch.delenv(DASHBOARD_TOKEN_ENV, raising=False)


def test_configured_token_is_none_when_unset() -> None:
    """No env var means auth is off."""
    assert configured_token() is None


def test_authorized_is_always_true_when_no_token_configured() -> None:
    """With no token configured, any (or no) credential passes."""
    assert authorized(None, None) is True
    assert authorized("whatever", None) is True


def test_authorized_checks_query_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured token gates on an exact query-param match."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    assert authorized(None, None) is False
    assert authorized("wrong", None) is False
    assert authorized("secret", None) is True


def test_authorized_checks_bearer_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured token also accepts a matching bearer header."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    assert authorized(None, "Bearer wrong") is False
    assert authorized(None, "Basic secret") is False
    assert authorized(None, "Bearer secret") is True


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


def test_download_bundle_401s_without_a_token_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured dashboard token gates the bundle route too."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    _save_checkpoint()
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(download_bundle(_NAME))
    assert excinfo.value.status_code == 401


def test_download_bundle_accepts_the_query_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bundle route accepts the token as a query param."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    _save_checkpoint()
    response = asyncio.run(download_bundle(_NAME, token="secret"))
    path = Path(response.path)
    with zipfile.ZipFile(path) as archive:
        assert bm.MANIFEST_NAME in archive.namelist()


def test_download_bundle_accepts_the_bearer_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bundle route also accepts an Authorization: Bearer header."""
    monkeypatch.setenv(DASHBOARD_TOKEN_ENV, "secret")
    _save_checkpoint()
    response = asyncio.run(
        download_bundle(
            _NAME, request=_FakeRequest({"authorization": "Bearer secret"}),
        )
    )
    assert Path(response.path).exists()
