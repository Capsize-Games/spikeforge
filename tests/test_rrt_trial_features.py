"""Tests for the RRT's query+context feature encoding (issue #26)."""

import torch

from spikeforge.rrt.relation import NUM_ENTITIES
from spikeforge.rrt.trial_features import feature_dim, trial_features

_QUERY_WIDTH = 2 * NUM_ENTITIES


def test_empty_history_is_the_all_zero_context() -> None:
    """No prior trials means an all-zero context summary."""
    features = trial_features(0, 1, [])
    context = features[_QUERY_WIDTH:]
    assert torch.equal(context, torch.zeros_like(context))


def test_feature_vector_has_the_declared_width() -> None:
    """The encoded feature vector always matches ``feature_dim()``."""
    features = trial_features(0, 1, [(1, 2, 0), (0, 2, 1)])
    assert features.shape == (feature_dim(),)


def test_repeating_a_trial_strengthens_its_evidence() -> None:
    """More repeats of the same fact saturate toward stronger evidence."""
    one_trial = trial_features(0, 1, [(0, 1, 1)])
    two_trials = trial_features(0, 1, [(0, 1, 1), (0, 1, 1)])
    assert two_trials.sum() > one_trial.sum()


def test_outcome_zero_and_flipped_slots_agree_on_the_winner() -> None:
    """(a, b, 1) and (b, a, 0) are the same fact and encode identically."""
    as_i_beats_j = trial_features(2, 0, [(0, 1, 1)])
    as_j_beats_i = trial_features(2, 0, [(1, 0, 0)])
    assert torch.equal(as_i_beats_j, as_j_beats_i)


def test_different_trials_change_the_context_summary() -> None:
    """A different history produces a different context summary."""
    first = trial_features(0, 1, [(0, 1, 1)])
    second = trial_features(0, 1, [(1, 2, 0)])
    assert not torch.allclose(first, second)
