"""Unexportable stage kinds fail with a typed, named error."""

import pytest

from snn_interpreter.nir_bridge import stages_unmappable, to_nir
from snn_interpreter.nir_bridge.errors import UnsupportedStageError
from snn_interpreter.nir_bridge.mapper import map_stage
from snn_interpreter.topology import registry
from snn_interpreter.topology.stage import Stage

pytest.importorskip("nir")

_KINDS = tuple(sorted(stages_unmappable.UNMAPPABLE_STAGES))


@pytest.mark.parametrize("kind", _KINDS)
def test_unexportable_kind_raises_with_named_reason(kind: str) -> None:
    """Each unexportable kind raises the typed error naming kind and reason."""
    with pytest.raises(UnsupportedStageError) as excinfo:
        map_stage(Stage("s", kind, {}))
    assert excinfo.value.kind == kind
    assert kind in str(excinfo.value)
    assert stages_unmappable.UNMAPPABLE_STAGES[kind] in str(excinfo.value)


def test_sequence_attn_export_names_first_unexportable_stage() -> None:
    """Exporting the attention demo fails on its first unexportable stage."""
    spec, module = registry.build_topology("sequence_attn")
    with pytest.raises(UnsupportedStageError) as excinfo:
        to_nir(spec, module)
    assert excinfo.value.kind == "embedding"


def test_alpha_neuron_still_raises_named_error() -> None:
    """The pre-existing ``alpha`` precedent is unchanged."""
    spec, module = registry.build_topology("fc_legacy", {"neuron": "alpha"})
    with pytest.raises(UnsupportedStageError) as excinfo:
        to_nir(spec, module)
    assert excinfo.value.kind == "alpha"


def test_unmappable_table_carries_reasons() -> None:
    """Every unexportable kind has a non-empty honest reason."""
    reasons = stages_unmappable.UNMAPPABLE_STAGES
    expected = {"embedding", "attention", "layer_norm", "maxpool2d"}
    assert expected <= set(reasons)
    assert all(reason for reason in reasons.values())
