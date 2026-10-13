"""The ``spikeforge-hub`` account subcommands."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from spikeforge_hub import account_cli, cache, cli, client
from spikeforge_hub.errors import HubApiError

_ACCOUNT: Dict[str, Any] = {
    "handle": "ada",
    "email": "ada@example.test",
    "storage": {"used_bytes": 3, "limit_bytes": 10},
}


class _FakeHub:
    """Stand-in HubClient answering with a canned account."""

    def __init__(self, token: Optional[str] = None) -> None:
        """Record the token this client was built with."""
        self.token = token

    def me(self) -> Dict[str, Any]:
        """Return the canned account."""
        return dict(_ACCOUNT)


class _RefusingHub(_FakeHub):
    """Stand-in HubClient refusing every call."""

    def me(self) -> Dict[str, Any]:
        """Refuse, as a revoked token would be."""
        raise HubApiError("/v1/me", "HTTP 401: bad token", 401)


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the hub cache into a temporary directory, logged out."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path / "hub"))
    monkeypatch.delenv(client.TOKEN_ENV, raising=False)
    return tmp_path


def test_login_stores_a_verified_token(
    sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A pasted token that verifies is stored and its handle printed."""
    monkeypatch.setattr(account_cli, "_read_pat", lambda: "secret-pat")
    monkeypatch.setattr(account_cli, "HubClient", _FakeHub)
    assert cli.main(["login"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["handle"] == "ada"
    assert Path(payload["stored"]).read_text().strip() == "secret-pat"


def test_login_refused_stores_nothing(
    sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A token the server refuses is reported and never written."""
    monkeypatch.setattr(account_cli, "_read_pat", lambda: "bad-pat")
    monkeypatch.setattr(account_cli, "HubClient", _RefusingHub)
    assert cli.main(["login"]) == 1
    assert "nothing stored" in capsys.readouterr().out
    assert not Path(client.token_path()).exists()


def test_login_blank_paste_stores_nothing(
    sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty paste exits without touching the API or the disk."""
    monkeypatch.setattr(account_cli, "_read_pat", lambda: "")
    assert cli.main(["login"]) == 1
    assert "login stores nothing" in capsys.readouterr().out


def test_whoami_prints_the_account(
    sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``whoami`` prints the profile and storage position as JSON."""
    monkeypatch.setattr(account_cli, "require_client", lambda: _FakeHub())
    assert cli.main(["whoami"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["handle"] == "ada"
    assert payload["storage"]["used_bytes"] == 3


def test_whoami_logged_out_names_login(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``whoami`` with no token names login instead of failing blankly."""
    assert cli.main(["whoami"]) == 1
    assert "not logged in" in capsys.readouterr().out


def test_quota_prints_only_the_storage_block(
    sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``quota`` renders the storage block, not the whole profile."""
    monkeypatch.setattr(account_cli, "require_client", lambda: _FakeHub())
    assert cli.main(["quota"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"used_bytes": 3, "limit_bytes": 10}


def test_push_passes_flags_to_transfer(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``push`` forwards its flags and prints the transfer report."""
    calls: List[Any] = []

    def fake_push(hub: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        calls.append((args, kwargs))
        return {"state": "published", "upload_id": "u1"}
    monkeypatch.setattr(account_cli, "require_client", lambda: _FakeHub())
    monkeypatch.setattr(account_cli, "push_saved_model", fake_push)
    argv = ["push", "ckpt", "--version", "2", "--license", "MIT",
           "--slug", "s", "--summary", "sum"]
    assert cli.main(argv) == 0
    args, kwargs = calls[0]
    assert args == ("ckpt", "2", "MIT")
    assert (kwargs["slug"], kwargs["summary"]) == ("s", "sum")
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "published"


def test_pull_gates_on_the_import_verdict(
    sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``pull`` exits non-zero for an incompatible import, like import."""
    seen: Dict[str, str] = {}

    def fake_pull(pull_id: str) -> Dict[str, Any]:
        seen["id"] = pull_id
        return {"import": {"verdict": {"verdict": "incompatible"}}}

    monkeypatch.setattr(account_cli, "pull_model", fake_pull)
    assert cli.main(["pull", "@ada/mnist-fc"]) == 1
    assert seen["id"] == "@ada/mnist-fc"
    payload = json.loads(capsys.readouterr().out)
    assert payload["import"]["verdict"]["verdict"] == "incompatible"
