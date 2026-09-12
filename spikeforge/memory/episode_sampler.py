"""N-way K-shot episode sampling for episodic metric-learning (issue #24)."""

import random
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

#: One sampled episode: support images/labels, then query images/labels.
Episode = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def class_indices(
    dataset: Dataset, classes: Sequence[int],
) -> Dict[int, List[int]]:
    """Return each of ``classes``'s sample indices within ``dataset``."""
    targets = dataset.targets
    return {
        cls: torch.nonzero(targets == cls).squeeze(1).tolist()
        for cls in classes
    }


def sample_episode(
    dataset: Dataset,
    by_class: Dict[int, List[int]],
    n_way: int,
    k_shot: int,
    n_query: int,
    rng: random.Random,
) -> Episode:
    """Sample one episode: ``n_way`` classes, ``k_shot`` + ``n_query`` each.

    Labels are episode-local (``0..n_way-1``), independent of the
    dataset's real class ids -- the point of episodic training is a
    task-agnostic "these match / these don't" signal, not a fixed
    N-way classifier.
    """
    chosen = rng.sample(list(by_class), n_way)
    support: Tuple[List[torch.Tensor], List[int]] = ([], [])
    query: Tuple[List[torch.Tensor], List[int]] = ([], [])
    for label, cls in enumerate(chosen):
        indices = rng.sample(by_class[cls], k_shot + n_query)
        _fill(dataset, indices[:k_shot], label, support)
        _fill(dataset, indices[k_shot:], label, query)
    return (
        torch.stack(support[0]), torch.tensor(support[1]),
        torch.stack(query[0]), torch.tensor(query[1]),
    )


def _fill(
    dataset: Dataset,
    indices: List[int],
    label: int,
    accumulator: Tuple[List[torch.Tensor], List[int]],
) -> None:
    """Append ``dataset``'s images at ``indices`` under one episode label."""
    images, labels = accumulator
    for index in indices:
        image, _ = dataset[index]
        images.append(image)
        labels.append(label)
