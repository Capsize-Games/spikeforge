"""Tests for the RRT's hierarchy/cycle relation (issue #26)."""

import random

from spikeforge.rrt.relation import (
    NUM_ENTITIES,
    cycle_outcome,
    hierarchy_outcome,
    sample_pair,
    sample_rank,
)

_RANK = (0, 1, 2)  # entity 0 = top, 1 = middle, 2 = bottom


def test_hierarchy_is_transitive() -> None:
    """Top beats middle, middle beats bottom, and top beats bottom."""
    assert hierarchy_outcome(_RANK, 0, 1) == 1
    assert hierarchy_outcome(_RANK, 1, 2) == 1
    assert hierarchy_outcome(_RANK, 0, 2) == 1


def test_hierarchy_outcomes_are_antisymmetric() -> None:
    """Every pair's outcome flips when the arguments are swapped."""
    for i in range(NUM_ENTITIES):
        for j in range(NUM_ENTITIES):
            if i != j:
                assert hierarchy_outcome(_RANK, i, j) != (
                    hierarchy_outcome(_RANK, j, i)
                )


def test_cycle_flips_only_the_top_bottom_edge() -> None:
    """The cycle keeps the chain edges and reverses only top-vs-bottom."""
    assert cycle_outcome(_RANK, 0, 1) == hierarchy_outcome(_RANK, 0, 1)
    assert cycle_outcome(_RANK, 1, 2) == hierarchy_outcome(_RANK, 1, 2)
    assert cycle_outcome(_RANK, 0, 2) != hierarchy_outcome(_RANK, 0, 2)


def test_cycle_is_non_transitive() -> None:
    """Top beats middle, middle beats bottom, but bottom beats top."""
    assert cycle_outcome(_RANK, 0, 1) == 1
    assert cycle_outcome(_RANK, 1, 2) == 1
    assert cycle_outcome(_RANK, 2, 0) == 1


def test_sample_rank_is_a_permutation() -> None:
    """A sampled rank assignment covers every rank exactly once."""
    rank = sample_rank(random.Random(0))
    assert sorted(rank) == list(range(NUM_ENTITIES))


def test_sample_pair_is_two_distinct_entities() -> None:
    """A sampled pair never repeats the same entity."""
    i, j = sample_pair(random.Random(0))
    assert i != j
    assert 0 <= i < NUM_ENTITIES
    assert 0 <= j < NUM_ENTITIES
