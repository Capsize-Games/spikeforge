"""The Relational Reversal Task falsification attempt (issue #26).

Meta-trains a frozen, context-conditioned predictor -- the
"strongest current challenge" category the Consciousness Gradient
paper names (a purely parametrically determined system, no local
plasticity, no online weight updates) -- on a random-permutation
family of 3-entity dominance hierarchies, then throws the paper's own
Relational Reversal Task at it: mid-session, silently, the hierarchy
is reversed into a non-transitive cycle, with no retraining and no
signal that anything changed except the feedback itself.

Run from the repository root::

    venv/bin/python examples/13_relational_reversal_task.py

Reports whichever way it lands, per the issue's own instruction: a
successful post-reversal recovery without retraining would falsify
the paper's A3 axiom; a failure to recover is a real data point for
it. Either way, compare the reversal gap against the non-reversed
control gap (prediction (c)) before reading anything into the raw
post-reversal number alone.
"""

from spikeforge.rrt.falsification_poc import run_falsification_attempt


def main() -> int:
    """Run the falsification attempt once and print the paper's metrics."""
    result = run_falsification_attempt()
    print(f"sessions: {result.sessions}")
    print(f"pre-reversal accuracy:  {result.pre_reversal_accuracy:.1%}")
    print(f"post-reversal accuracy: {result.post_reversal_accuracy:.1%}")
    print(f"reversal gap:           {result.reversal_gap:.1%}")
    print(f"control pre accuracy:   {result.control_pre_accuracy:.1%}")
    print(f"control post accuracy:  {result.control_post_accuracy:.1%}")
    print(f"control (no-reversal) gap: {result.control_gap:.1%}")
    print(f"sessions that recovered: {result.recovered_fraction:.1%}")
    if result.mean_trials_to_recovery is not None:
        print(
            f"mean trials-to-recovery (when recovered): "
            f"{result.mean_trials_to_recovery:.1f}"
        )
    else:
        print("mean trials-to-recovery: no session recovered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
