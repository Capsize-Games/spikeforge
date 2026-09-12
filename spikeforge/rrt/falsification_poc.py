"""Aggregate the RRT falsification attempt over many sessions (issue #26).

Phase 1 scope: meta-train the frozen, context-conditioned predictor
from :mod:`spikeforge.rrt.context_model`, then run it through many
independent reversal sessions and an equal number of non-reversed
control sessions (the within-distribution comparison the paper's
prediction (c) calls for), and report the paper's own metrics.
"""

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional

from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.rrt.context_model import build_frozen_predictor
from spikeforge.rrt.protocol import SessionResult, run_one_session
from spikeforge.topology.stage_module import StageModule


@dataclass(frozen=True)
class FalsificationResult:
    """Aggregated pre/post accuracy and recovery across many sessions."""

    pre_reversal_accuracy: float
    post_reversal_accuracy: float
    control_pre_accuracy: float
    control_post_accuracy: float
    recovered_fraction: float
    mean_trials_to_recovery: Optional[float]
    sessions: int

    @property
    def reversal_gap(self) -> float:
        """Return the accuracy drop caused specifically by reversal."""
        return self.pre_reversal_accuracy - self.post_reversal_accuracy

    @property
    def control_gap(self) -> float:
        """Return the accuracy drop in the non-reversed control."""
        return self.control_pre_accuracy - self.control_post_accuracy


def run_falsification_attempt(
    seed: int = 0,
    train_episodes: int = 600,
    hidden: int = 32,
    num_steps: int = 15,
    sessions: int = 100,
    pre_trials: int = 60,
    post_trials: int = 60,
) -> FalsificationResult:
    """Meta-train a frozen predictor, then run the RRT and its control."""
    encoder = SpikeEncoder(coding="rate", num_steps=num_steps)
    net = build_frozen_predictor(encoder, train_episodes, hidden, seed=seed)
    reversed_results = _run_sessions(
        net, encoder, sessions, pre_trials, post_trials, True, seed + 1,
    )
    control_results = _run_sessions(
        net, encoder, sessions, pre_trials, post_trials, False, seed + 2,
    )
    return _summarize(reversed_results, control_results)


def _run_sessions(
    net: StageModule,
    encoder: SpikeEncoder,
    sessions: int,
    pre_trials: int,
    post_trials: int,
    reverse: bool,
    seed: int,
) -> List[SessionResult]:
    """Run ``sessions`` independent interaction sessions."""
    rng = random.Random(seed)
    return [
        run_one_session(net, encoder, pre_trials, post_trials, reverse, rng)
        for _ in range(sessions)
    ]


def _mean(values: Iterable[float]) -> float:
    """Return the mean of ``values``, or 0.0 when empty."""
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _summarize(
    reversed_results: List[SessionResult],
    control_results: List[SessionResult],
) -> FalsificationResult:
    """Reduce raw sessions into the paper's own reported metrics."""
    recovered = [
        r.trials_to_recovery for r in reversed_results
        if r.trials_to_recovery is not None
    ]
    return FalsificationResult(
        pre_reversal_accuracy=_mean(
            r.pre_asymptote for r in reversed_results
        ),
        post_reversal_accuracy=_mean(
            r.post_asymptote for r in reversed_results
        ),
        control_pre_accuracy=_mean(r.pre_asymptote for r in control_results),
        control_post_accuracy=_mean(
            r.post_asymptote for r in control_results
        ),
        recovered_fraction=len(recovered) / len(reversed_results),
        mean_trials_to_recovery=_mean(recovered) if recovered else None,
        sessions=len(reversed_results),
    )
