"""PT-W4 acceptance: the ``spikeforge-clients`` CLI.

Covers ``info``, ``predict`` (JSON and ``.npy`` files), ``stream`` (stdin),
and the typed failure exits. The CLI's client is routed through the
dependency-free in-process ASGI transport, so no port is bound.
"""

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_serve_app import _sample, _write_bundle

from spikeforge_clients import cli
from spikeforge_clients.client import ServeClient
from spikeforge_clients.errors import ServiceError
from spikeforge_serve import create_app


@pytest.fixture()
def app(tmp_path: Path) -> Any:
    """Return the in-process serve app over a fresh bundle."""
    path = tmp_path / "model.spkf"
    _write_bundle(path)
    return create_app(str(path))


@pytest.fixture(autouse=True)
def route_through_the_app(monkeypatch: pytest.MonkeyPatch, app: Any) -> None:
    """Point the CLI's client at the in-process ASGI app."""
    monkeypatch.setattr(
        cli, "_client", lambda args: ServeClient.in_process(app)
    )


def _json_file(path: Path, document: Any) -> str:
    """Write ``document`` as JSON and return its path."""
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def test_info_prints_bundle_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``info`` prints the bundle metadata as JSON and exits zero."""
    assert cli.main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["topology"] == "fc_small"
    assert payload["num_classes"] == 3


def test_predict_reads_frames_from_a_json_file(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """``predict`` loads a JSON frame list and prints the response."""
    path = _json_file(tmp_path / "frames.json", [_sample(1)])
    assert cli.main(["predict", "--file", path]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["steps"] == 4
    assert payload["predictions"][0]["label"] in range(3)


def test_predict_reads_frames_from_an_npy_file(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """``predict`` loads a ``.npy`` array and prints the response."""
    path = tmp_path / "frames.npy"
    np.save(path, np.asarray([_sample(2)]))
    assert cli.main(["predict", "--file", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["steps"] == 4
    assert len(payload["predictions"]) == 1


def test_stream_reads_frames_from_stdin(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``stream`` reads one JSON frame per line and prints JSON lines."""
    lines = json.dumps(_sample(3)) + "\n" + json.dumps(_sample(4)) + "\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(lines))
    assert cli.main(["stream"]) == 0
    replies = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line
    ]
    assert [reply["type"] for reply in replies] == [
        "prediction",
        "prediction",
    ]
    assert replies[0]["payload"]["steps"] == 4
    assert replies[1]["payload"]["steps"] == 8


def test_predict_missing_file_exits_nonzero(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A missing frame file is a typed failure with exit code 1."""
    missing = str(tmp_path / "absent.json")
    assert cli.main(["predict", "--file", missing]) == 1
    assert "no such file" in capsys.readouterr().err.lower()


def test_predict_empty_frames_exit_nonzero(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """An empty frame list is refused before any request is made."""
    path = _json_file(tmp_path / "empty.json", [])
    assert cli.main(["predict", "--file", path]) == 1
    assert "non-empty" in capsys.readouterr().err


def test_service_error_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    app: Any,
) -> None:
    """A typed service error is reported and exits 1."""

    class _Failing(ServeClient):
        """A client whose bundle call always fails."""

        def bundle_info(self) -> Any:
            """Raise the typed service error the CLI must report."""
            raise ServiceError(404, "bundle_not_found", "missing")

    monkeypatch.setattr(
        cli, "_client", lambda args: _Failing.in_process(app)
    )
    assert cli.main(["info"]) == 1
    assert "bundle_not_found" in capsys.readouterr().err


def test_predict_requires_a_file() -> None:
    """Omitting ``--file`` is an argparse usage error (exit 2)."""
    with pytest.raises(SystemExit) as error:
        cli.main(["predict"])
    assert error.value.code == 2
