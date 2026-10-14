"""Per-kind verdicts for the NIR-unmappable stage record."""

from pathlib import Path
from typing import Dict, Tuple

import pytest

from spikeforge.nir_bridge import stages_unmappable
from spikeforge.nir_bridge.errors import (
    UNMAPPABLE_DECISIONS,
    UnsupportedStageError,
)
from spikeforge.nir_bridge.mapper import map_stage
from spikeforge.nir_bridge.neuron_nodes import UNMAPPABLE_KINDS
from spikeforge.topology.stage import Stage

_ROOT = Path(__file__).resolve().parent.parent
_DECISIONS_DOC = _ROOT / "plans" / "nir-unmappable-decisions.md"

#: The only verdicts the record may assign.
_VERDICTS = frozenset(
    {
        "await-upstream-primitive",
        "decompose-into-existing-NIR",
        "permanently-simulation-only",
    }
)


def _rows() -> Dict[str, Tuple[str, str, str]]:
    """Return the decision table keyed by stage kind."""
    rows: Dict[str, Tuple[str, str, str]] = {}
    in_table = False
    for line in _DECISIONS_DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Stage kind |"):
            in_table = True
            continue
        if in_table and line.startswith("| `"):
            cells = [cell.strip() for cell in line.split("|")]
            rows[cells[1].strip("`")] = (cells[2], cells[3], cells[4])
        elif in_table and not line.startswith("|"):
            break
    return rows


def _expected_kinds() -> Dict[str, str]:
    """Return every unmappable stage and neuron kind with its reason."""
    return {**stages_unmappable.UNMAPPABLE_STAGES, **UNMAPPABLE_KINDS}


def test_every_unmappable_kind_has_a_verdict() -> None:
    """Each unmappable kind owns exactly one table row, none missing."""
    assert set(_rows()) == set(_expected_kinds())


def test_every_verdict_is_decided_with_a_reason() -> None:
    """Verdicts use the allowed vocabulary and no row defers the call."""
    text = _DECISIONS_DOC.read_text(encoding="utf-8")
    assert "tbd" not in text.lower()
    for kind, (verdict, reason, _) in _rows().items():
        assert verdict in _VERDICTS, kind
        assert reason, kind


def test_decomposable_kinds_name_their_follow_up() -> None:
    """Decompositions point at the filed export issue by number."""
    rows = _rows()
    assert rows["embedding"][0] == "decompose-into-existing-NIR"
    assert rows["batch_norm"][0] == "decompose-into-existing-NIR"
    for kind, (verdict, _, follow_up) in rows.items():
        if verdict == "decompose-into-existing-NIR":
            assert "#" in follow_up, kind


@pytest.mark.parametrize("kind", sorted(_expected_kinds()))
def test_error_points_at_the_decision_record(kind: str) -> None:
    """The typed error for each kind references the record and its hint."""
    with pytest.raises(UnsupportedStageError) as excinfo:
        map_stage(Stage("stuck", kind, {}))
    assert excinfo.value.kind == kind
    assert UNMAPPABLE_DECISIONS in str(excinfo.value)
    assert excinfo.value.hint == UNMAPPABLE_DECISIONS


def test_unknown_kind_carries_no_record_pointer() -> None:
    """Kinds outside the record fail without pointing at it."""
    with pytest.raises(UnsupportedStageError) as excinfo:
        map_stage(Stage("stuck", "no_such_kind", {}))
    assert excinfo.value.hint == ""
    assert UNMAPPABLE_DECISIONS not in str(excinfo.value)
