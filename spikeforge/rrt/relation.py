"""A 3-entity hierarchy and its reversal into a non-transitive cycle.

Deliberately not tied to the vision datasets in #22/#24 -- a fresh,
abstract task, per the issue's own requirement.
"""

import random
from typing import Tuple

#: Number of abstract entities in the toy relational structure.
NUM_ENTITIES = 3
#: An entity-id -> rank assignment (0 = top); a random one is a "session".
Rank = Tuple[int, int, int]


def sample_rank(rng: random.Random) -> Rank:
    """Return a random top/middle/bottom assignment over the 3 entities."""
    ranks = list(range(NUM_ENTITIES))
    rng.shuffle(ranks)
    return (ranks[0], ranks[1], ranks[2])


def hierarchy_outcome(rank: Rank, i: int, j: int) -> int:
    """Return 1 if ``i`` beats ``j`` under the transitive hierarchy.

    The higher-ranked (lower ``rank`` value) entity always wins -- the
    unique transitive tournament on 3 entities.
    """
    return int(rank[i] < rank[j])


def cycle_outcome(rank: Rank, i: int, j: int) -> int:
    """Return 1 if ``i`` beats ``j`` after reversal into a 3-cycle.

    Keeps the hierarchy's top-beats-middle and middle-beats-bottom
    edges intact and flips only the top-vs-bottom edge -- the single,
    minimal change that turns a transitive order into the classic
    non-transitive cycle (top > middle > bottom > top).
    """
    top = rank.index(0)
    bottom = rank.index(NUM_ENTITIES - 1)
    if {i, j} == {top, bottom}:
        return int(i == bottom)
    return hierarchy_outcome(rank, i, j)


def sample_pair(rng: random.Random) -> Tuple[int, int]:
    """Return a random ordered pair of distinct entity ids."""
    i, j = rng.sample(range(NUM_ENTITIES), 2)
    return i, j
