"""The nir-c-runtime capability profile and its single-sourced kinds."""

import json
from pathlib import Path
from typing import Set

from spikeforge_targets import c_runtime
from spikeforge_targets.c_runtime_errors import CRuntimeUnsupportedError
from spikeforge_targets.c_runtime_report import CRuntimeReport
from spikeforge_targets.c_runtime_violation import CRuntimeViolation

_ROOT = Path(__file__).resolve().parent.parent
_PROFILE_DOC = _ROOT / "plans" / "nir-c-runtime-profile.md"

#: The C loader's KIND_TABLE, in graph.h order, as a regression pin.
_KIND_TABLE = frozenset(
    {
        "Input",
        "Output",
        "Conv2d",
        "Affine",
        "Flatten",
        "AvgPool2d",
        "SumPool2d",
        "LI",
        "Threshold",
        "Scale",
        "Delay",
    }
)


def _doc_kinds() -> Set[str]:
    """Return the backticked kinds in the profile doc's support table."""
    kinds: Set[str] = set()
    in_table = False
    for line in _PROFILE_DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("| NIR kind |"):
            in_table = True
            continue
        if in_table and line.startswith("| `"):
            kinds.add(line.split("`")[1])
        elif in_table and not line.startswith("|"):
            break
    return kinds


def test_supported_matches_the_c_kind_table() -> None:
    """The profile pins exactly the 11 kinds the C loader implements."""
    assert c_runtime.SUPPORTED == _KIND_TABLE


def test_spec_declares_the_profile_honestly() -> None:
    """The spec names the target, its kinds, and float32/no-quant limits."""
    spec = c_runtime.spec()
    assert spec.name == c_runtime.TARGET_NAME == "nir-c-runtime"
    assert spec.supported == c_runtime.SUPPORTED
    assert spec.constraints["dtype"] == "float32"
    assert spec.constraints["quantization"] == "none"
    assert spec.constraints["activation_quantization"] == "none"
    assert json.dumps(spec.to_dict())


def test_profile_doc_tables_the_single_sourced_kinds() -> None:
    """The doc's support table equals the constant, tracking a commit."""
    assert _doc_kinds() == set(c_runtime.SUPPORTED)
    text = _PROFILE_DOC.read_text(encoding="utf-8")
    assert c_runtime.TRACKED_COMMIT in text
    assert "spikeforge_targets/c_runtime.py" in text


def test_report_marks_a_clean_export_passed() -> None:
    """A report without violations passes and serialises to JSON."""
    report = CRuntimeReport(
        target=c_runtime.TARGET_NAME,
        supported=("input", "output"),
        unsupported=(),
    )
    assert report.passed() is True
    assert report.counts() == {
        "supported": 2,
        "unsupported": 0,
        "total": 2,
    }
    assert json.dumps(report.to_dict())


def test_typed_error_names_each_violation() -> None:
    """The refusal carries names, kinds, and reasons without parsing."""
    violations = (
        CRuntimeViolation("lif1", "LIF", "unsupported node kind 'LIF'"),
        CRuntimeViolation("fc", "Linear", "unsupported node kind 'Linear'"),
    )
    error = CRuntimeUnsupportedError(violations)
    assert error.names == ("lif1", "fc")
    assert error.kinds == ("LIF", "Linear")
    assert error.violations == violations
    assert "'lif1' (LIF)" in str(error)
    assert "'fc' (Linear)" in str(error)
