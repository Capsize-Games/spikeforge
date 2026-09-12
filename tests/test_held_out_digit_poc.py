"""Smoke test for the held-out-digit protocol's wiring (issue #22).

Uses a tiny subset and few time steps so it runs in CI seconds; it is
a wiring check (does the base-frozen + one-shot-memory pipeline run
and stay internally consistent), not an accuracy benchmark. The full
protocol is exercised for real accuracy numbers in
``examples/11_held_out_digit_memory.py``.
"""

import copy

import torch
from torch.utils.data import DataLoader

from spikeforge.data.datasets import build_dataset
from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.memory.frozen_classifier import build_frozen_classifier
from spikeforge.memory.held_out_digit_poc import (
    HELD_OUT_DIGIT,
    digit_subset,
    run_held_out_digit_poc,
)


def test_digit_subset_excludes_the_held_out_digit() -> None:
    """Filtering to digits 0-8 never yields a digit-9 label."""
    train_data = build_dataset("mnist", train=True)
    subset = digit_subset(train_data, range(HELD_OUT_DIGIT))
    labels = {int(train_data.targets[i]) for i in subset.indices}
    assert HELD_OUT_DIGIT not in labels


def test_freezing_blocks_further_optimizer_steps() -> None:
    """A frozen classifier's parameters do not move under a backward pass."""
    train_data = build_dataset("mnist", train=True)
    loader = DataLoader(
        digit_subset(train_data, range(HELD_OUT_DIGIT)),
        batch_size=8, shuffle=False,
    )
    encoder = SpikeEncoder(coding="rate", num_steps=3)
    net = build_frozen_classifier(loader, encoder, hidden=8, num_classes=9)
    before = copy.deepcopy(list(net.parameters()))
    for param in net.parameters():
        assert not param.requires_grad
    assert all(
        torch.equal(a, b) for a, b in zip(before, net.parameters())
    )


def test_run_held_out_digit_poc_reports_both_numbers() -> None:
    """The end-to-end protocol yields two accuracy percentages."""
    result = run_held_out_digit_poc(
        hidden=8, num_steps=3, epochs=1,
        batch_size=32, eval_batch_size=32,
    )
    assert 0.0 <= result.retained_accuracy <= 100.0
    assert 0.0 <= result.recall_accuracy <= 100.0
