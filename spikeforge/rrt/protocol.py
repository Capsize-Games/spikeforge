"""The RRT falsification protocol: pre-reversal, reversal, post-reversal.

Implements the paper's own operationalization (Section 9.1/9.2): a
session of pairwise interactions, fed to a frozen predictor whose
weights never move, with a silent mid-session relation change and no
explicit training-mode boundary ever exposed to it.
"""

import random
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.rrt.relation import (
    Rank,
    cycle_outcome,
    hierarchy_outcome,
    sample_pair,
    sample_rank,
)
from spikeforge.rrt.trial_features import Trial, trial_features
from spikeforge.simulator.runner import run
from spikeforge.topology.stage_module import StageModule

#: How many of a phase's trailing trials define its "asymptotic" accuracy.
ASYMPTOTE_WINDOW = 10
#: Recovery tolerance: within this fraction of pre-reversal accuracy.
RECOVERY_TOLERANCE = 0.1

Relation = Callable[[Rank, int, int], int]


@dataclass
class SessionResult:
    """One session's trial-by-trial correctness, split by phase."""

    pre_reversal: List[int]
    post_reversal: List[int]

    @property
    def pre_asymptote(self) -> float:
        """Return accuracy over the pre-reversal phase's last trials."""
        return _tail_mean(self.pre_reversal, ASYMPTOTE_WINDOW)

    @property
    def post_asymptote(self) -> float:
        """Return accuracy over the post-reversal phase's last trials."""
        return _tail_mean(self.post_reversal, ASYMPTOTE_WINDOW)

    @property
    def trials_to_recovery(self) -> Optional[int]:
        """Return the first post-reversal trial count back near baseline.

        ``None`` means recovery never happened within the session.
        """
        target = self.pre_asymptote - RECOVERY_TOLERANCE
        for end in range(ASYMPTOTE_WINDOW, len(self.post_reversal) + 1):
            window = self.post_reversal[:end]
            if _tail_mean(window, ASYMPTOTE_WINDOW) >= target:
                return end
        return None


def _tail_mean(trials: List[int], window: int) -> float:
    """Return the mean of ``trials``'s last ``window`` entries."""
    tail = trials[-window:] if trials else []
    return sum(tail) / len(tail) if tail else 0.0


def run_one_session(
    net: StageModule,
    encoder: SpikeEncoder,
    pre_trials: int,
    post_trials: int,
    reverse: bool,
    rng: random.Random,
) -> SessionResult:
    """Run one interaction session, optionally reversing the relation.

    The model's weights never change; only the growing ``history`` it
    is queried with does. When ``reverse`` is False this is the
    within-distribution control: the session continues under its
    original hierarchy for ``post_trials`` more trials instead.
    """
    rank = sample_rank(rng)
    history: List[Trial] = []
    pre = _run_phase(
        net, encoder, rank, hierarchy_outcome, pre_trials, history, rng,
    )
    relation = cycle_outcome if reverse else hierarchy_outcome
    post = _run_phase(
        net, encoder, rank, relation, post_trials, history, rng,
    )
    return SessionResult(pre, post)


def _run_phase(
    net: StageModule,
    encoder: SpikeEncoder,
    rank: Rank,
    relation: Relation,
    num_trials: int,
    history: List[Trial],
    rng: random.Random,
) -> List[int]:
    """Run ``num_trials`` queries under ``relation``, growing ``history``."""
    correctness = []
    for _ in range(num_trials):
        i, j = sample_pair(rng)
        prediction = _predict(net, encoder, i, j, history)
        outcome = relation(rank, i, j)
        correctness.append(int(prediction == outcome))
        history.append((i, j, outcome))
    return correctness


def _predict(
    net: StageModule,
    encoder: SpikeEncoder,
    i: int,
    j: int,
    history: List[Trial],
) -> int:
    """Return the frozen predictor's prediction for one query."""
    feature = trial_features(i, j, history).unsqueeze(0)
    with torch.no_grad():
        trajectory = run(net, encoder.encode(feature))
    return int(trajectory.logits.argmax(dim=1).item())
