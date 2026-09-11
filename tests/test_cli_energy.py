"""The ``spikeforge-energy`` CLI subcommands and their JSON output."""

import json
from pathlib import Path

import pytest

from spikeforge_targets.energy import cli

_SMALL = ["--topology", "fc_small", "--steps", "2", "--batch-size", "1"]


def test_account_prints_an_estimated_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``account --sparse`` prints a declared-cost estimate."""
    assert cli.main(["account", *_SMALL, "--target", "reference",
                     "--sparse"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["basis"] == "declared cost table"
    assert payload["estimate"] is True
    assert payload["energy"] is not None
    assert payload["ops"]["sop"] < payload["ops"]["mac"]


def test_account_without_a_table_reports_unavailable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A target with no declared table reports unavailable, not a number."""
    assert cli.main(["account", *_SMALL, "--target", "no_such_target",
                     "--sparse"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["basis"] == "unavailable"
    assert payload["energy"] is None
    assert payload["latency"] is None


def test_report_writes_the_parity_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``report --out`` writes the report plus a sparse-vs-dense comparison."""
    out = tmp_path / "energy.json"
    args = ["report", "--topology", "conv_net", "--target", "lava_loihi2",
            "--sparse", "--steps", "2", "--batch-size", "1",
            "--out", str(out)]
    assert cli.main(args) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    assert payload["report"]["target"] == "lava_loihi2"
    assert comparison["within_tolerance"] is True
    assert comparison["max_abs"] <= comparison["tolerance"]
    capsys.readouterr()
