"""Declared per-target cost tables: loading, validation, and lookup."""

from pathlib import Path

import pytest

from spikeforge_targets.energy import target_costs
from spikeforge_targets.energy.cost_table import REQUIRED_KEYS, CostTable
from spikeforge_targets.energy.errors import EnergyCostError


def test_bundled_tables_load_and_are_declared() -> None:
    """Every shipped table is valid and explicitly not measured."""
    names = target_costs.names()
    for expected in ("reference", "norse", "lava_loihi2", "spinnaker2"):
        assert expected in names
    for name in names:
        table = target_costs.load(name)
        assert table is not None
        assert table.measured is False
        assert table.sop_pj > 0.0
        assert table.step_ns > 0.0
        assert table.source


def test_unknown_target_has_no_table() -> None:
    """A target without a bundled table is reported absent, not guessed."""
    assert target_costs.load("no_such_target") is None
    assert target_costs.declared("no_such_target") is False


def test_from_dict_requires_every_key() -> None:
    """A table missing a required key raises a typed error."""
    with pytest.raises(EnergyCostError) as excinfo:
        CostTable.from_dict("x", {"sop_pj": 1.0})
    assert excinfo.value.target == "x"
    for key in ("mac_pj", "ac_pj", "step_ns", "source", "measured"):
        assert key in excinfo.value.detail


def test_energy_and_latency_math() -> None:
    """The per-op totals add up and the dense total is reported separately."""
    table = target_costs.load("reference")
    assert table is not None
    energy = table.energy_pj(10, 100, 5)
    assert energy["sop_pj"] == 10 * table.sop_pj
    assert energy["mac_pj"] == 100 * table.mac_pj
    assert energy["ac_pj"] == 5 * table.ac_pj
    assert energy["total_pj"] == energy["sop_pj"] + energy["ac_pj"]
    assert energy["dense_pj"] == energy["mac_pj"] + energy["ac_pj"]
    latency = table.latency_ns(4)
    assert latency == {"step_ns": table.step_ns, "total_ns": 4 * table.step_ns}


def test_malformed_table_raises_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed JSON table is a named failure, not a bare traceback."""
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(target_costs, "COSTS_DIR", tmp_path)
    target_costs._tables.cache_clear()
    try:
        with pytest.raises(EnergyCostError):
            target_costs.load("broken")
    finally:
        target_costs._tables.cache_clear()


def test_required_keys_are_declared() -> None:
    """The validation key set is the documented contract."""
    assert set(REQUIRED_KEYS) == {
        "sop_pj", "mac_pj", "ac_pj", "step_ns", "source", "measured",
    }
