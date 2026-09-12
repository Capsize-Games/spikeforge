"""Tests for the prototypical-network loss (issue #24)."""

import torch

from spikeforge.memory.prototypical import prototypical_loss


def test_perfect_clusters_score_perfect_accuracy() -> None:
    """Query points placed exactly at their prototype are always right."""
    support_emb = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    support_labels = torch.tensor([0, 1])
    query_emb = torch.tensor([[0.1, 0.0], [9.9, 10.0]])
    query_labels = torch.tensor([0, 1])
    _, accuracy = prototypical_loss(
        support_emb, support_labels, query_emb, query_labels, n_way=2,
    )
    assert accuracy == 1.0


def test_swapped_queries_score_zero_accuracy() -> None:
    """A query nearest the wrong prototype is scored wrong."""
    support_emb = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    support_labels = torch.tensor([0, 1])
    query_emb = torch.tensor([[0.1, 0.0], [9.9, 10.0]])
    query_labels = torch.tensor([1, 0])
    _, accuracy = prototypical_loss(
        support_emb, support_labels, query_emb, query_labels, n_way=2,
    )
    assert accuracy == 0.0


def test_loss_is_finite_and_decreases_with_tighter_clusters() -> None:
    """A tighter, better-separated embedding yields a lower loss."""
    support_labels = torch.tensor([0, 1])
    query_labels = torch.tensor([0, 1])
    loose, _ = prototypical_loss(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]), support_labels,
        torch.tensor([[0.4, 0.4], [0.6, 0.6]]), query_labels, n_way=2,
    )
    tight, _ = prototypical_loss(
        torch.tensor([[0.0, 0.0], [10.0, 10.0]]), support_labels,
        torch.tensor([[0.1, 0.1], [9.9, 9.9]]), query_labels, n_way=2,
    )
    assert torch.isfinite(loose) and torch.isfinite(tight)
    assert tight < loose
