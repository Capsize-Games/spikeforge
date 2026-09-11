"""Third-party PyTorch extraction via ``nirtorch`` (Phase F3).

The real extraction path runs when ``nir``/``nirtorch`` are installed; the
honest-absence path hides ``nirtorch`` so the typed named error is covered
without the extra. No test silently drops an unsupported node.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from snn_interpreter.cli import verify
from snn_interpreter.nir_bridge import (
    api,
    extract,
    extract_summary,
    run_extracted,
)
from snn_interpreter.nir_bridge.errors import (
    ExtractionExtraMissingError,
    UnsupportedNodeError,
)

pytest.importorskip("nir")


def _sequential() -> torch.nn.Module:
    """Return a small hand-built two-layer dense module."""
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))


def _kinds(graph: Any) -> list:
    """Return the node kinds of ``graph`` in insertion order."""
    return [type(node).__name__ for node in graph.nodes.values()]


def test_extract_lifts_a_sequential_to_nir() -> None:
    """A plain ``nn.Sequential`` extracts to an Input/Affine/Output graph."""
    pytest.importorskip("nirtorch")
    kinds = _kinds(extract(_sequential()))
    assert kinds.count("Affine") == 2
    assert kinds[0] == "Input"
    assert kinds[-1] == "Output"


def test_extracted_graph_runs_on_the_interpreter() -> None:
    """The extracted graph executes independently and yields a readout."""
    pytest.importorskip("nirtorch")
    result = run_extracted(_sequential(), torch.rand(5, 2, 4))
    assert result.steps == 5
    assert result.readout.shape[-1] == 2


def test_extract_summary_names_nodes_and_edges() -> None:
    """The summary is JSON-able and carries the extracted nodes and edges."""
    pytest.importorskip("nirtorch")
    summary = extract_summary(_sequential())
    assert json.dumps(summary)
    assert [node["kind"] for node in summary["nodes"]].count("Affine") == 2
    assert len(summary["edges"]) == 3


def test_unsupported_module_raises_named_error() -> None:
    """A module with no faithful NIR node names the offending class."""
    pytest.importorskip("nirtorch")
    module = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
    with pytest.raises(UnsupportedNodeError) as excinfo:
        extract(module)
    assert excinfo.value.kind == "ReLU"


def test_convolution_is_reported_not_dropped() -> None:
    """A conv module is refused by name rather than silently truncated."""
    pytest.importorskip("nirtorch")
    with pytest.raises(UnsupportedNodeError) as excinfo:
        extract(torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3)))
    assert excinfo.value.kind == "Conv2d"


def test_missing_nirtorch_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the extra, extraction reports it instead of raising KeyError."""
    monkeypatch.setitem(sys.modules, "nirtorch", None)
    with pytest.raises(ExtractionExtraMissingError) as ctx:
        extract(_sequential())
    assert ctx.value.extra == "nir"


def test_capability_reports_nirtorch_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hiding nirtorch flips the probe honestly, without raising."""
    monkeypatch.setitem(sys.modules, "nirtorch", None)
    report = api.capability()
    assert report["nirtorch_available"] is False


def _saved(tmp_path: Path, module: torch.nn.Module, name: str) -> str:
    """Save ``module`` into ``tmp_path`` and return the path."""
    path = tmp_path / name
    torch.save(module, path)
    return str(path)


def test_cli_extract_prints_a_runnable_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``snn-targets extract`` prints the report and exits zero."""
    pytest.importorskip("nirtorch")
    args = ["extract", "--module", _saved(tmp_path, _sequential(), "ok.pt")]
    assert verify.main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runnable"] is True
    assert payload["features"] == 4


def test_cli_extract_names_an_unsupported_node(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unmappable module prints the typed reason and exits non-zero."""
    pytest.importorskip("nirtorch")
    module = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
    args = ["extract", "--module", _saved(tmp_path, module, "bad.pt")]
    assert verify.main(args) == 1
    assert "ReLU" in capsys.readouterr().out


def test_cli_extract_missing_file_is_a_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing module file becomes a message, not a traceback."""
    assert verify.main(["extract", "--module", "missing.pt"]) == 1
    assert capsys.readouterr().out.strip()
