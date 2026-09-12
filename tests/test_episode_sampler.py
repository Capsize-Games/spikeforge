"""Tests for N-way K-shot episode sampling (issue #24)."""

import random

import torch
from torch.utils.data import TensorDataset

from spikeforge.memory.episode_sampler import class_indices, sample_episode


def _toy_dataset() -> TensorDataset:
    """Return a 4-class toy dataset with 10 samples per class."""
    images = torch.arange(40).float().view(40, 1)
    targets = torch.arange(40) // 10
    dataset = TensorDataset(images, targets)
    dataset.targets = targets
    return dataset


def test_class_indices_groups_by_label() -> None:
    """Each class's indices all carry that class's target label."""
    dataset = _toy_dataset()
    by_class = class_indices(dataset, [0, 1, 2, 3])
    for cls, indices in by_class.items():
        assert all(int(dataset.targets[i]) == cls for i in indices)


def test_sample_episode_shapes_and_labels() -> None:
    """An episode has the requested shot/query counts and local labels."""
    dataset = _toy_dataset()
    by_class = class_indices(dataset, [0, 1, 2, 3])
    support_x, support_y, query_x, query_y = sample_episode(
        dataset, by_class, n_way=3, k_shot=2, n_query=4,
        rng=random.Random(0),
    )
    assert support_x.shape[0] == 3 * 2
    assert query_x.shape[0] == 3 * 4
    assert set(support_y.tolist()) == {0, 1, 2}
    assert set(query_y.tolist()) == {0, 1, 2}


def test_episode_labels_are_disjoint_from_dataset_labels() -> None:
    """Episode labels are always ``0..n_way-1``, not the real class ids."""
    dataset = _toy_dataset()
    by_class = class_indices(dataset, [0, 1, 2, 3])
    _, support_y, _, query_y = sample_episode(
        dataset, by_class, n_way=2, k_shot=1, n_query=1,
        rng=random.Random(1),
    )
    assert set(support_y.tolist()) | set(query_y.tolist()) <= {0, 1}
