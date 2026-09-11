"""CLI save/list/compare flow and its regression exit code."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from spikeforge.benchmark.__main__ import main
from spikeforge.benchmark.store import BenchmarkStore


def _args(store: Path, *extra: str) -> List[str]:
    """Return a fast CPU fixture with ``extra`` arguments appended."""
    return [
        "--topology", "fc_small",
        "--device", "cpu",
        "--steps", "3",
        "--repeats", "1",
        "--warmup", "0",
        "--no-backward",
        "--store-dir", str(store),
        *extra,
    ]


def _record(ms_per_step: float, steps_per_second: float) -> Dict[str, Any]:
    """Return a one-config record with the given throughput."""
    return {
        "results": [
            {
                "topology": "fc_small",
                "mode": "production",
                "forward": {
                    "mean_ms_per_step": ms_per_step,
                    "steps_per_second": steps_per_second,
                },
                "memory": {"tracemalloc_peak_bytes": 1},
            }
        ]
    }


def test_cli_save_then_list_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--save`` persists a run that ``--list`` then reports."""
    assert main(_args(tmp_path, "--save", "--label", "ci")) == 0
    capsys.readouterr()
    assert main(_args(tmp_path, "--list")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["label"] == "ci"


def test_cli_compare_against_stored_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--compare`` diffs a baseline and a stored candidate as JSON."""
    store = BenchmarkStore(str(tmp_path))
    baseline = store.save(_record(1.0, 1000.0), label="base")
    candidate = store.save(_record(1.0, 1000.0), label="cand")
    code = main(
        _args(tmp_path, "--compare", baseline, "--against", candidate)
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["compared"] == 1
    assert result["regressed"] is False


def test_cli_compare_exits_nonzero_on_requested_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--fail-on-regression`` mirrors :func:`exit_code` when it regresses."""
    store = BenchmarkStore(str(tmp_path))
    baseline = store.save(_record(1.0, 1000.0), label="base")
    candidate = store.save(_record(10.0, 100.0), label="slow")
    code = main(
        _args(
            tmp_path,
            "--compare", baseline,
            "--against", candidate,
            "--fail-on-regression",
        )
    )
    assert code == 1
    capsys.readouterr()
