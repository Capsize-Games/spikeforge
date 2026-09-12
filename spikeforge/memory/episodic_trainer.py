"""Episodic (metric-learning) training loop for a spiking embedder."""

import random
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.memory.episode_sampler import class_indices, sample_episode
from spikeforge.memory.prototypical import embed, prototypical_loss
from spikeforge.topology.registry import build_topology
from spikeforge.topology.stage_module import StageModule

#: Learning rate for the embedder's Adam optimizer.
LEARNING_RATE = 1e-3


def build_episodic_embedder(
    dataset: Dataset,
    classes: Sequence[int],
    encoder: SpikeEncoder,
    embed_dim: int,
    hidden: int,
    episodes: int,
    n_way: int,
    k_shot: int,
    n_query: int,
    seed: int = 0,
) -> StageModule:
    """Train ``fc_small`` as an embedder via prototypical-network episodes.

    Unlike ``fc_small``'s ordinary use as a fixed-class classifier,
    the final layer here carries no softmax semantics: episodic
    training only ever asks it to place same-class spike patterns
    close together and different classes apart, which is what should
    let it generalise -- to classes held out of every episode, and in
    the fuller claim this issue tests, to a script never trained on
    at all (see ``few_shot_generalization_poc.py``).
    """
    _, net = build_topology(
        "fc_small", {"hidden": hidden, "num_classes": embed_dim},
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    rng = random.Random(seed)
    by_class = class_indices(dataset, classes)
    for _ in range(episodes):
        _train_one_episode(
            net, optimizer, dataset, by_class, encoder,
            n_way, k_shot, n_query, rng,
        )
    return net


def _train_one_episode(
    net: StageModule,
    optimizer: torch.optim.Optimizer,
    dataset: Dataset,
    by_class: Dict[int, List[int]],
    encoder: SpikeEncoder,
    n_way: int,
    k_shot: int,
    n_query: int,
    rng: random.Random,
) -> None:
    """Sample one episode and take one prototypical-loss gradient step."""
    support_x, support_y, query_x, query_y = sample_episode(
        dataset, by_class, n_way, k_shot, n_query, rng,
    )
    support_emb = embed(net, encoder, support_x)
    query_emb = embed(net, encoder, query_x)
    loss, _ = prototypical_loss(
        support_emb, support_y, query_emb, query_y, n_way,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
