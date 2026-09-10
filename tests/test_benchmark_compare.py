"""Regression detection for the benchmark comparator."""

from typing import Any, Dict

from snn_interpreter.benchmark.compare import (
    DEFAULT_THRESHOLD,
    compare_runs,
    exit_code,
)


def _record(
    topology: str,
    mode: str,
    ms_per_step: float,
    steps_per_second: float,
    memory: int,
) -> Dict[str, Any]:
    """Return a minimal benchmark record for the comparator."""
    return {
        "topology": topology,
        "mode": mode,
        "forward": {
            "mean_ms_per_step": ms_per_step,
            "steps_per_second": steps_per_second,
        },
        "memory": {"tracemalloc_peak_bytes": memory},
    }


def _report(
    ms_per_step: float, steps_per_second: float, memory: int
) -> Dict[str, Any]:
    """Return a one-config report with the given throughput and memory."""
    return {
        "results": [
            _record(
                "fc_small", "production", ms_per_step, steps_per_second,
                memory,
            )
        ]
    }


def test_compare_flags_a_regression() -> None:
    """A slower run past the threshold is flagged and gates non-zero."""
    baseline = _report(10.0, 100.0, 1000)
    candidate = _report(13.0, 77.0, 1000)
    result = compare_runs(baseline, candidate, threshold=0.1)
    assert result["regressed"] is True
    assert result["regressions"] == ["fc_small:production"]
    assert exit_code(result) == 1


def test_compare_tolerates_change_within_threshold() -> None:
    """Small throughput wobble inside the threshold is not a regression."""
    baseline = _report(10.0, 100.0, 1000)
    candidate = _report(10.5, 95.0, 1000)
    result = compare_runs(baseline, candidate, threshold=0.1)
    assert result["regressed"] is False
    assert result["regressions"] == []
    assert exit_code(result) == 0


def test_compare_flags_memory_growth() -> None:
    """Peak memory growing past the threshold is a regression."""
    baseline = _report(10.0, 100.0, 1000)
    candidate = _report(10.0, 100.0, 2000)
    result = compare_runs(baseline, candidate, threshold=0.1)
    assert result["regressed"] is True


def test_compare_skips_unmatched_configs() -> None:
    """Only configs present in both reports are compared."""
    baseline = _report(10.0, 100.0, 1000)
    other = {
        "results": [_record("conv_net", "production", 5.0, 200.0, 1)]
    }
    result = compare_runs(baseline, other)
    assert result["compared"] == 0
    assert result["regressed"] is False


def test_default_threshold_is_ten_percent() -> None:
    """The documented default threshold is ten percent."""
    assert DEFAULT_THRESHOLD == 0.1
