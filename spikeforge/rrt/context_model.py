"""A frozen, meta-trained, context-conditioned predictor (issue #26).

Trained once via backprop across many randomly permuted hierarchy
sessions, then frozen. Its only test-time adaptation mechanism is the
input itself changing as :mod:`spikeforge.rrt.trial_features`
recomputes the context summary from a growing history -- no weight
update, no local plasticity: exactly the "strongest current
challenge" category the paper's benchmark names.
"""

import random
from typing import List, Tuple

import torch

from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.rrt.relation import (
    Rank,
    hierarchy_outcome,
    sample_pair,
    sample_rank,
)
from spikeforge.rrt.trial_features import Trial, feature_dim, trial_features
from spikeforge.simulator.runner import run
from spikeforge.topology.registry import build_topology
from spikeforge.topology.stage_module import StageModule

#: Learning rate for the predictor's (offline, one-time) Adam training.
LEARNING_RATE = 1e-2
#: Longest history sampled during meta-training.
MAX_TRAIN_CONTEXT = 12


def build_frozen_predictor(
    encoder: SpikeEncoder,
    episodes: int,
    hidden: int = 32,
    batch_size: int = 32,
    seed: int = 0,
) -> StageModule:
    """Meta-train a predictor over many random hierarchy permutations.

    Every training example comes from the *hierarchy* family only,
    never the reversed cycle -- the frozen predictor must never see a
    reversal before the evaluation protocol springs it on it. Each
    example resamples both the session's rank permutation and the
    context length, so the only way to do well is to infer the
    current session's hierarchy from its context, not memorize one.
    """
    _, net = build_topology(
        "fc_small",
        {"hidden": hidden, "num_classes": 2, "input_size": feature_dim()},
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    rng = random.Random(seed)
    for _ in range(episodes):
        _train_batch(net, optimizer, encoder, batch_size, rng)
    net.requires_grad_(False)
    net.eval()
    return net


def _train_batch(
    net: StageModule,
    optimizer: torch.optim.Optimizer,
    encoder: SpikeEncoder,
    batch_size: int,
    rng: random.Random,
) -> None:
    """Sample one batch of (context, query, label) and take a step."""
    examples = [_sample_training_example(rng) for _ in range(batch_size)]
    features = torch.stack([feature for feature, _ in examples])
    labels = torch.tensor([label for _, label in examples])
    trajectory = run(net, encoder.encode(features))
    loss = torch.nn.functional.cross_entropy(trajectory.logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _sample_training_example(
    rng: random.Random,
) -> Tuple[torch.Tensor, int]:
    """Sample one (feature vector, label) pair from a fresh session."""
    rank = sample_rank(rng)
    context_len = rng.randint(0, MAX_TRAIN_CONTEXT)
    history: List[Trial] = [
        _random_trial(rank, rng) for _ in range(context_len)
    ]
    i, j = sample_pair(rng)
    label = hierarchy_outcome(rank, i, j)
    return trial_features(i, j, history), label


def _random_trial(rank: Rank, rng: random.Random) -> Trial:
    """Sample one past ``(i, j, observed outcome)`` trial under ``rank``."""
    i, j = sample_pair(rng)
    return i, j, hierarchy_outcome(rank, i, j)
