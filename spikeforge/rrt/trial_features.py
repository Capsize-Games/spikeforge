"""Turn a query pair and interaction history into one input vector."""

from typing import List, Tuple

import torch

from spikeforge.rrt.relation import NUM_ENTITIES

#: One completed trial: (entity i, entity j, observed outcome for i-vs-j).
Trial = Tuple[int, int, int]
#: Evidence strength saturates once a pair has been seen this many times.
WIN_SATURATION = 3.0


def feature_dim() -> int:
    """Return the input vector width :func:`trial_features` produces."""
    query_dim = 2 * NUM_ENTITIES
    context_dim = NUM_ENTITIES * NUM_ENTITIES
    return query_dim + context_dim


def _one_hot(index: int, size: int) -> torch.Tensor:
    """Return a length-``size`` one-hot vector with ``index`` set."""
    vector = torch.zeros(size)
    vector[index] = 1.0
    return vector


def _query_vector(i: int, j: int) -> torch.Tensor:
    """Return the query pair's one-hot encoding, ``[2 * NUM_ENTITIES]``."""
    return torch.cat([
        _one_hot(i, NUM_ENTITIES), _one_hot(j, NUM_ENTITIES),
    ])


def _context_summary(history: List[Trial]) -> torch.Tensor:
    """Encode observed outcomes as a slot-invariant win-count matrix.

    Cell ``(a, b)`` saturates toward 1.0 as more trials directly show
    ``a`` beating ``b``, regardless of whether a trial was recorded as
    ``(a, b, 1)`` or ``(b, a, 0)`` -- both are the same evidence. A
    naive average of the raw ``(i, j, outcome)`` one-hots (an earlier
    version of this function) pools each slot's marginal separately
    and cannot recover "who beat whom" once more than one trial is in
    context; this matrix keeps every trial's winner/loser association
    intact, which is what let the predictor actually learn (see
    plans/memory_system_research.md's issue #26 section).
    """
    wins = torch.zeros(NUM_ENTITIES, NUM_ENTITIES)
    for i, j, outcome in history:
        winner, loser = (i, j) if outcome == 1 else (j, i)
        wins[winner, loser] += 1.0
    return (wins / WIN_SATURATION).clamp(max=1.0).flatten()


def trial_features(
    i: int, j: int, history: List[Trial],
) -> torch.Tensor:
    """Return the frozen predictor's input for querying ``(i, j)``.

    Concatenates the query pair's one-hot encoding with a win-count
    summary of every trial observed so far in this session (both pre-
    and post-reversal, exactly matching the "no explicit training-mode
    boundary" requirement).
    """
    return torch.cat([_query_vector(i, j), _context_summary(history)])
