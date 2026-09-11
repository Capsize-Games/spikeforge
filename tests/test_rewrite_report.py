"""Rewrite report shape, readiness, and JSON safety."""

import json

import numpy as np
import pytest

from spikeforge.nir_bridge import api
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology import presets
from spikeforge_targets.rewrite import rewrite
from spikeforge_targets.rewrite_report import RewriteReport

pytest.importorskip("nir")

_CONV = presets.conv_net(
    in_channels=1, channels=2, num_classes=3, input_size=8
)


def _conv_graph() -> object:
    """Return a small ``Input -> Conv2d -> Output`` graph."""
    source = api.node_class("Input")({"input": None})
    sink = api.node_class("Output")({"output": None})
    weight = np.zeros((1, 1, 3, 3), np.float32)
    conv = api.node_class("Conv2d")(
        None, weight, 1, 0, 1, 1, np.zeros(1, np.float32)
    )
    return api.node_class("NIRGraph")(
        {"input": source, "conv": conv, "output": sink},
        [("input", "conv"), ("conv", "output")],
        type_check=False,
    )


def test_reference_rewrite_changes_nothing_and_is_ready() -> None:
    """The reference target needs no rewrite and reports itself ready."""
    report = rewrite(to_nir(_CONV), "reference").report
    assert report.applied == ()
    assert report.skipped == ()
    assert report.unfixable == ()
    assert report.rewritten is False
    assert report.ready() is True


def test_report_buckets_are_json_serialisable() -> None:
    """A report with an applied and an unfixable node serialises."""
    report = rewrite(to_nir(_CONV), "lava_loihi2").report
    payload = report.to_dict()
    assert json.dumps(payload)
    assert payload["counts"] == {
        "applied": 1,
        "skipped": 0,
        "unfixable": 0,
    }


def test_unfixable_report_is_not_ready() -> None:
    """A graph with an unfixable node is not ready to compile."""
    result = rewrite(_conv_graph(), "xylo")
    assert result.report.ready() is False
    assert result.ready() is False
    assert json.dumps(result.report.to_dict())
    assert result.report.to_dict()["drift"] is None


def test_report_counts_track_every_bucket() -> None:
    """The counts mirror the record lists exactly."""
    report = RewriteReport(
        target="t",
        applied=({"node": "a"},),
        skipped=({"node": "b"},),
        unfixable=({"node": "c"},),
        rewritten=True,
    )
    assert report.counts() == {
        "applied": 1,
        "skipped": 1,
        "unfixable": 1,
    }
    assert report.ready() is False
