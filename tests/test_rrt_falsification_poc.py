"""Smoke test for the RRT falsification protocol's wiring (issue #26).

Tiny settings so it runs in CI seconds; it is a wiring check, not the
actual falsification attempt. Real numbers come from
``examples/13_relational_reversal_task.py``.
"""

from spikeforge.rrt.falsification_poc import run_falsification_attempt


def test_run_reports_bounded_metrics() -> None:
    """The protocol yields accuracies in [0, 1] and a sane session count."""
    result = run_falsification_attempt(
        train_episodes=3, hidden=4, num_steps=3,
        sessions=2, pre_trials=12, post_trials=12,
    )
    assert 0.0 <= result.pre_reversal_accuracy <= 1.0
    assert 0.0 <= result.post_reversal_accuracy <= 1.0
    assert 0.0 <= result.control_pre_accuracy <= 1.0
    assert 0.0 <= result.control_post_accuracy <= 1.0
    assert 0.0 <= result.recovered_fraction <= 1.0
    assert result.sessions == 2
