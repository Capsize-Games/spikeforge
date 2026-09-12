"""Tests for the SNN-native one-shot associative memory (issue #22)."""

import torch

from spikeforge.memory.one_shot_associative_memory import (
    OneShotAssociativeMemory,
)

_PATTERN_A = torch.tensor(
    [[1.0, 1.0, 0.0, 0.0]] * 5,
)
_PATTERN_B = torch.tensor(
    [[0.0, 0.0, 1.0, 1.0]] * 5,
)


def _run(memory: OneShotAssociativeMemory, spike_train: torch.Tensor) -> int:
    """Drive ``memory`` over ``spike_train`` and return total spikes."""
    memory.reset_state(batch_size=1)
    total = 0
    for step in spike_train:
        total += int(memory.step(step.unsqueeze(0)).sum())
    return total


def test_untaught_memory_never_fires() -> None:
    """With no taught classes, recall spikes are structurally zero."""
    memory = OneShotAssociativeMemory(in_features=4)
    assert _run(memory, _PATTERN_A) == 0


def test_taught_pattern_is_recalled() -> None:
    """The example a class was taught on fires more than a novel one."""
    memory = OneShotAssociativeMemory(in_features=4)
    memory.teach(_PATTERN_A)
    taught = _run(memory, _PATTERN_A)
    novel = _run(memory, _PATTERN_B)
    assert taught > novel


def test_teaching_a_second_class_does_not_erase_the_first() -> None:
    """A second one-shot write leaves the first class's recall intact."""
    memory = OneShotAssociativeMemory(in_features=4)
    memory.teach(_PATTERN_A)
    before = _run(memory, _PATTERN_A)
    memory.teach(_PATTERN_B)
    after = _run(memory, _PATTERN_A)
    assert after == before


def test_num_classes_tracks_teach_calls() -> None:
    """Each ``teach`` call grows the class count by exactly one."""
    memory = OneShotAssociativeMemory(in_features=4)
    assert memory.num_classes == 0
    memory.teach(_PATTERN_A)
    assert memory.num_classes == 1
    memory.teach(_PATTERN_B)
    assert memory.num_classes == 2
