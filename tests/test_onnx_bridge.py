"""ONNX import/export bridge (Phase F2).

Real export/import runs when the optional ``onnx`` extra is present; the
absent path hides ``onnx`` so the typed named error is covered without it, and
the orchestration tests stub :mod:`snn_interpreter.onnx_bridge.api` so the
present path is exercised even when the extra is missing. An unmappable op is
always refused by name rather than silently dropped.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch

from snn_interpreter.cli import verify
from snn_interpreter.onnx_bridge import (
    api,
    export,
    import_onnx,
    metadata,
    roundtrip,
)
from snn_interpreter.onnx_bridge.errors import (
    OnnxExtraMissingError,
    UnsupportedOnnxOpError,
)
from snn_interpreter.topology.registry import build_topology

# --- metadata round-trip (no onnx needed) ---------------------------------


def test_metadata_roundtrips_a_spec() -> None:
    """The encoded spec decodes back to an identical topology."""
    spec, _ = build_topology("conv_net")
    values = metadata.encode(spec, "conv_net")
    assert metadata.topology_of(values) == "conv_net"
    assert metadata.decode(values).to_dict() == spec.to_dict()


def test_metadata_refuses_unusable_payloads() -> None:
    """An absent or malformed spec decodes to ``None``, never an exception."""
    assert metadata.decode({}) is None
    assert metadata.decode({metadata.SPEC_KEY: "{not json"}) is None


# --- absent extra: honest, named path -------------------------------------


def _hide_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the onnx stack appear absent for the duration of a test."""
    monkeypatch.setitem(sys.modules, "onnx", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)


def test_capability_survives_absent_onnx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hiding onnx flips the probe honestly, without raising."""
    _hide_onnx(monkeypatch)
    report = api.capability()
    assert report["onnx_available"] is False
    assert report["onnxruntime_available"] is False


def test_export_names_the_missing_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without the extra, export raises the typed error naming it."""
    _hide_onnx(monkeypatch)
    with pytest.raises(OnnxExtraMissingError) as ctx:
        export.export_topology("conv_net", str(tmp_path / "x.onnx"))
    assert ctx.value.extra == "onnx"


def test_import_names_the_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the extra, import raises the typed error naming it."""
    _hide_onnx(monkeypatch)
    with pytest.raises(OnnxExtraMissingError):
        import_onnx.import_report("missing.onnx")


# --- orchestration with a stubbed isolated API ----------------------------


def _patch_io(
    monkeypatch: pytest.MonkeyPatch,
    model: Any,
    sink: Dict[str, Any],
    fake_export: Any,
) -> None:
    """Install the stubbed isolated-API surface used by the export path."""
    monkeypatch.setattr(api, "export_onnx", fake_export)
    monkeypatch.setattr(api, "load_onnx", lambda path: model)
    monkeypatch.setattr(
        api, "save_onnx", lambda m, path: sink.update({"save": path})
    )
    monkeypatch.setattr(
        api, "set_metadata", lambda m, v: sink.update({"values": v})
    )
    monkeypatch.setattr(api, "op_types", lambda m: ["Gemm"])


def _fake_model(monkeypatch: pytest.MonkeyPatch, sink: Dict[str, Any]) -> Any:
    """Patch the isolated API with an in-memory ONNX model stub."""
    model = SimpleNamespace(
        graph=SimpleNamespace(node=[object()]), metadata_props=[]
    )

    def fake_export(
        module: Any, args: Any, path: str, opset: int = 17
    ) -> None:
        sink["export"] = {"path": path, "opset": opset}

    _patch_io(monkeypatch, model, sink, fake_export)
    return model


def test_export_built_writes_and_stamps_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The export path writes the model, stamps metadata, and reports ops."""
    sink: Dict[str, Any] = {}
    _fake_model(monkeypatch, sink)
    spec, module = build_topology("fc_small")
    target = str(tmp_path / "fc.onnx")
    report = export.export_built(spec, module, target, "fc_small")
    assert sink["export"]["path"] == target
    assert sink["values"][metadata.TOPOLOGY_KEY] == "fc_small"
    assert report["ops"] == ["Gemm"]
    assert report["temporal"] == metadata.TEMPORAL_CONTRACT


def test_import_report_maps_stubbed_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The op-mapping path builds a spec without any real onnx file."""
    descriptor = {
        "op_type": "Gemm",
        "name": "dense",
        "inputs": ["x", "w"],
        "attrs": {"transB": 1},
        "weight_shape": [2, 4],
    }
    monkeypatch.setattr(api, "load_onnx", lambda path: object())
    monkeypatch.setattr(api, "metadata_values", lambda m: {})
    monkeypatch.setattr(api, "graph_ops", lambda m: [descriptor])
    monkeypatch.setattr(api, "op_types", lambda m: ["Gemm"])
    report = import_onnx.import_report("third-party.onnx")
    assert report["source"] == "ops"
    assert report["stages"] == [{"name": "dense", "kind": "linear"}]


def test_unsupported_op_is_refused_by_name() -> None:
    """A descriptor with no faithful stage raises a typed named error."""
    descriptor = {
        "op_type": "Softmax",
        "name": "act",
        "inputs": ["x"],
        "attrs": {},
        "weight_shape": None,
    }
    with pytest.raises(UnsupportedOnnxOpError) as excinfo:
        import_onnx._stage_for(descriptor, 0)
    assert excinfo.value.op_type == "Softmax"


# --- real ONNX path (skipped only when the extra is absent) ---------------


def test_conv_net_exports_and_reimports(tmp_path: Path) -> None:
    """``conv_net`` exports and re-imports to an identical topology."""
    pytest.importorskip("onnx")
    report = roundtrip("conv_net", str(tmp_path / "conv.onnx"))
    assert report["identical"] is True
    assert report["source"] == "metadata"
    assert "Conv" in report["ops"]


def test_fc_small_exports_and_reimports(tmp_path: Path) -> None:
    """A flat-entry topology also round-trips exactly through ONNX."""
    pytest.importorskip("onnx")
    report = roundtrip("fc_small", str(tmp_path / "fc.onnx"))
    assert report["identical"] is True


def test_third_party_ops_map_to_stages(tmp_path: Path) -> None:
    """A model with no metadata maps its ops to sized stage kinds."""
    pytest.importorskip("onnx")
    module = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    target = str(tmp_path / "third.onnx")
    api.export_onnx(module, (torch.rand(1, 4),), target)
    spec = import_onnx.spec_from_file(target)
    assert [stage.kind for stage in spec.stages] == ["linear", "linear"]
    assert spec.stages[0].params["in_features"] == 4


def test_unmappable_op_is_refused_by_name(tmp_path: Path) -> None:
    """An exported ``Relu`` op is refused by name, never dropped."""
    pytest.importorskip("onnx")
    target = str(tmp_path / "relu.onnx")
    api.export_onnx(torch.nn.ReLU(), (torch.rand(1, 4),), target)
    with pytest.raises(UnsupportedOnnxOpError) as excinfo:
        import_onnx.spec_from_file(target)
    assert excinfo.value.op_type == "Relu"


# --- CLI ------------------------------------------------------------------


def test_cli_onnx_export_and_import(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``onnx-export`` then ``onnx-import`` print JSON and exit zero."""
    pytest.importorskip("onnx")
    target = str(tmp_path / "cli.onnx")
    assert verify.main(["onnx-export", "--out", target]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["topology"] == "conv_net"
    assert verify.main(["onnx-import", "--file", target]) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["source"] == "metadata"


def test_cli_onnx_roundtrip_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``onnx-roundtrip`` reports fidelity and exits zero when identical."""
    pytest.importorskip("onnx")
    target = str(tmp_path / "trip.onnx")
    args = ["onnx-roundtrip", "--topology", "fc_small", "--out", target]
    assert verify.main(args) == 0
    assert json.loads(capsys.readouterr().out)["identical"] is True


def test_cli_onnx_export_names_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing extra becomes a message plus a non-zero exit."""
    _hide_onnx(monkeypatch)
    target = str(tmp_path / "nope.onnx")
    assert verify.main(["onnx-export", "--out", target]) == 1
    assert "onnx" in capsys.readouterr().out
