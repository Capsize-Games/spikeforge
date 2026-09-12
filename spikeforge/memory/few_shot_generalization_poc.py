"""Few-shot generalization to never-trained characters (issue #24).

Trains a spiking embedder episodically on EMNIST letters (English
script), freezes it, then tests whether it supports one-shot
discrimination among KMNIST (Japanese Kuzushiji) characters -- a
script the embedder never saw in any form during training. The
readout is #22's unchanged
:class:`~spikeforge.memory.one_shot_associative_memory.\
OneShotAssociativeMemory`: one memory neuron per class in the
evaluation episode, taught from a single support example and scored
on held-out queries.

This is a genuinely harder and uncertain claim than issue #22: no
amount of clever memory retrieval helps distinguish two classes the
embedder was never shown anything resembling; only a spike-pattern
embedding that generalises across scripts can.
"""

import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from spikeforge.data.datasets import build_dataset
from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.memory.embedding_readout import (
    memory_predictions,
    teach_from_example,
)
from spikeforge.memory.episode_sampler import class_indices, sample_episode
from spikeforge.memory.episodic_trainer import (
    DEFAULT_TOPOLOGY,
    build_episodic_embedder,
)
from spikeforge.memory.one_shot_associative_memory import (
    OneShotAssociativeMemory,
)
from spikeforge.topology.stage_module import StageModule

#: The script the embedder is trained on (English letters, 26 classes).
#: torchvision's EMNIST "letters" split labels classes 1-26, not 0-25.
TRAIN_DATASET = "emnist_letters"
TRAIN_CLASSES = tuple(range(1, 27))
#: The script it is evaluated on -- never seen during training.
EVAL_DATASET = "kmnist"
EVAL_CLASSES = tuple(range(10))


@dataclass(frozen=True)
class FewShotResult:
    """Mean one-shot discrimination accuracy over held-out episodes."""

    accuracy: float
    chance: float
    episodes: int


def run_few_shot_generalization_poc(
    seed: int = 0,
    hidden: int = 64,
    embed_dim: int = 32,
    num_steps: int = 15,
    train_episodes: int = 300,
    train_n_way: int = 5,
    train_k_shot: int = 5,
    train_n_query: int = 5,
    eval_episodes: int = 50,
    eval_n_way: int = 5,
    eval_n_query: int = 5,
    topology: str = DEFAULT_TOPOLOGY,
) -> FewShotResult:
    """Train on EMNIST letters, evaluate one-shot recall on KMNIST."""
    torch.manual_seed(seed)
    encoder = SpikeEncoder(coding="rate", num_steps=num_steps)
    net = _train_embedder(
        encoder, hidden, embed_dim, train_episodes,
        train_n_way, train_k_shot, train_n_query, seed, topology,
    )
    accuracy = _evaluate_cross_script(
        net, encoder, embed_dim, eval_episodes, eval_n_way,
        eval_n_query, seed,
    )
    return FewShotResult(accuracy, 1.0 / eval_n_way, eval_episodes)


def _train_embedder(
    encoder: SpikeEncoder,
    hidden: int,
    embed_dim: int,
    episodes: int,
    n_way: int,
    k_shot: int,
    n_query: int,
    seed: int,
    topology: str,
) -> StageModule:
    """Episodically train and freeze the embedder on EMNIST letters."""
    train_data = build_dataset(TRAIN_DATASET, train=True)
    net = build_episodic_embedder(
        train_data, TRAIN_CLASSES, encoder, embed_dim, hidden,
        episodes, n_way, k_shot, n_query, seed, topology,
    )
    net.requires_grad_(False)
    net.eval()
    return net


def _evaluate_cross_script(
    net: StageModule,
    encoder: SpikeEncoder,
    embed_dim: int,
    eval_episodes: int,
    n_way: int,
    n_query: int,
    seed: int,
) -> float:
    """Average one-shot discrimination accuracy over held-out episodes."""
    eval_data = build_dataset(EVAL_DATASET, train=False)
    by_class = class_indices(eval_data, EVAL_CLASSES)
    rng = random.Random(seed + 1)
    scores = [
        _run_one_episode(
            net, encoder, eval_data, by_class, embed_dim, n_way,
            n_query, rng,
        )
        for _ in range(eval_episodes)
    ]
    return sum(scores) / len(scores)


def _run_one_episode(
    net: StageModule,
    encoder: SpikeEncoder,
    eval_data: Dataset,
    by_class: Dict[int, List[int]],
    embed_dim: int,
    n_way: int,
    n_query: int,
    rng: random.Random,
) -> float:
    """Teach one memory neuron per class from one example; score queries."""
    support_x, support_y, query_x, query_y = sample_episode(
        eval_data, by_class, n_way, k_shot=1, n_query=n_query, rng=rng,
    )
    memory = OneShotAssociativeMemory(in_features=embed_dim)
    with torch.no_grad():
        for label in range(n_way):
            example = support_x[support_y == label]
            teach_from_example(memory, net, encoder.encode(example))
        preds = memory_predictions(memory, net, encoder.encode(query_x))
    return float((preds == query_y).float().mean())
