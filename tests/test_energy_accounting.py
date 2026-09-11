"""Accounting op counts to a declared estimate, or an honest absence."""

import pytest

from spikeforge_targets.energy import (
    BASIS_MEASURED,
    BASIS_TABLE,
    BASIS_UNAVAILABLE,
    account,
    measure_topology,
    probe,
    target_costs,
)

_COUNTS = {"sop": 100, "mac": 400, "ac": 50, "timesteps": 5}


def test_declared_table_yields_an_estimate() -> None:
    """A target with a table maps counts to an estimated energy/latency."""
    report = account(_COUNTS, "reference")
    table = target_costs.load("reference")
    assert table is not None
    assert report.estimate is True
    assert report.basis == BASIS_TABLE
    assert report.timesteps == 5
    assert report.ops == {"sop": 100, "mac": 400, "ac": 50}
    assert report.efficiency["sop_over_mac"] == pytest.approx(0.25)
    assert report.energy["total_pj"] == 100 * table.sop_pj + 50 * table.ac_pj
    assert report.latency["total_ns"] == 5 * table.step_ns
    assert any("estimate" in note for note in report.notes)


def test_missing_table_reports_unavailable_never_a_number() -> None:
    """A target without a table reports unavailable and no fabricated value."""
    report = account(_COUNTS, "no_such_target")
    assert report.estimate is True
    assert report.basis == BASIS_UNAVAILABLE
    assert report.energy is None
    assert report.latency is None
    assert report.efficiency["sop_over_mac"] == pytest.approx(0.25)
    assert any("no declared cost table" in note for note in report.notes)


def test_measurement_labels_the_report_measured() -> None:
    """A device-reported measurement flips estimate off and adds a block."""
    measurement = {
        "device": "fake-loihi", "total_pj": 12.5, "total_ns": 3000.0,
    }
    report = account(_COUNTS, "reference", measurement=measurement)
    assert report.estimate is False
    assert report.basis == BASIS_MEASURED
    assert report.energy == {"total_pj": 12.5}
    assert report.latency == {"total_ns": 3000.0}
    assert report.measured == measurement
    assert report.notes == ("measured on fake-loihi",)


def test_probe_reports_no_device_on_this_machine() -> None:
    """The isolated probe never fabricates a measurement."""
    assert probe.measure("lava_loihi2") is None
    assert "declared estimate" in probe.reason("lava_loihi2")


def test_dense_baseline_sets_sop_equal_to_mac() -> None:
    """``sparse=False`` reports the dense baseline (SOP == MAC)."""
    _, report = measure_topology(
        "fc_small", "reference", steps=2, batch=1, sparse=False
    )
    assert report.ops["sop"] == report.ops["mac"]


def test_sparse_account_reduces_sop() -> None:
    """The default sparse account reports fewer synaptic ops than MACs."""
    _, report = measure_topology(
        "fc_small", "reference", steps=2, batch=1, sparse=True
    )
    assert report.ops["sop"] < report.ops["mac"]
