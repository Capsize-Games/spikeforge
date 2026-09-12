"""Held-out-digit no-forgetting protocol (issue #22).

Trains ``fc_small`` on MNIST digits 0-8, freezes it, then one-shot
teaches digit 9 to a genuinely spiking
:class:`~spikeforge.memory.one_shot_associative_memory.\
OneShotAssociativeMemory` (Hebbian synapse + LIF readout, no
backprop). Reports two numbers: retained accuracy on 0-8 (should be
unchanged, since the base net is frozen) and recall accuracy on the
held-out 9 (learned from a single example).
"""

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import torch
from snntorch import utils
from torch.utils.data import DataLoader, Dataset, Subset

from spikeforge.data.datasets import build_dataset
from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.memory.frozen_classifier import (
    build_frozen_classifier,
    hidden_size,
)
from spikeforge.memory.lockstep_runner import (
    combined_predictions,
    hidden_trace,
)
from spikeforge.memory.one_shot_associative_memory import (
    OneShotAssociativeMemory,
)
from spikeforge.topology.stage_module import StageModule

#: The digit withheld from base training and taught one-shot instead.
HELD_OUT_DIGIT = 9
#: Digits 0-8: what the base classifier is trained to recognise.
BASE_DIGITS: Tuple[int, ...] = tuple(range(HELD_OUT_DIGIT))


@dataclass(frozen=True)
class HeldOutDigitResult:
    """Accuracy numbers from one held-out-digit POC run."""

    retained_accuracy: float
    recall_accuracy: float


def digit_subset(dataset: Dataset, digits: Sequence[int]) -> Subset:
    """Return the samples of ``dataset`` whose label is in ``digits``."""
    targets = dataset.targets
    mask = torch.isin(targets, torch.tensor(list(digits)))
    indices = torch.nonzero(mask).squeeze(1).tolist()
    return Subset(dataset, indices)


def run_held_out_digit_poc(
    seed: int = 0,
    hidden: int = 64,
    num_steps: int = 25,
    epochs: int = 1,
    batch_size: int = 64,
    eval_batch_size: int = 256,
    train_subset: int = 1,
) -> HeldOutDigitResult:
    """Run the full protocol once and return its two accuracy numbers.

    ``train_subset`` (see :func:`_train_base`) never touches the
    held-out digit or either evaluation set.
    """
    torch.manual_seed(seed)
    encoder = SpikeEncoder(coding="rate", num_steps=num_steps)
    train_data = build_dataset("mnist", train=True)
    test_data = build_dataset("mnist", train=False)

    net = _train_base(
        train_data, encoder, hidden, epochs, batch_size, train_subset,
    )
    memory = _teach_held_out_digit(net, train_data, encoder)
    return _evaluate_both(
        net, memory, test_data, encoder, eval_batch_size,
    )


def _evaluate_both(
    net: StageModule,
    memory: OneShotAssociativeMemory,
    test_data: Dataset,
    encoder: SpikeEncoder,
    eval_batch_size: int,
) -> HeldOutDigitResult:
    """Score retained (0-8) and recall (9) accuracy in one pass."""
    label_map = _label_mapper(memory)
    retained = _accuracy(
        net, memory, digit_subset(test_data, BASE_DIGITS),
        encoder, label_map, eval_batch_size,
    )
    recall = _accuracy(
        net, memory, digit_subset(test_data, (HELD_OUT_DIGIT,)),
        encoder, label_map, eval_batch_size,
    )
    return HeldOutDigitResult(retained, recall)


def _train_base(
    train_data: Dataset,
    encoder: SpikeEncoder,
    hidden: int,
    epochs: int,
    batch_size: int,
    train_subset: int,
) -> StageModule:
    """Train and freeze the base classifier on digits 0-8 only.

    ``train_subset`` reduces the training set by that integer factor
    via ``snntorch.utils.data_subset``, the same convention used
    elsewhere in this repo (e.g. ``SSNTrainer``); 1 keeps it whole.
    """
    base_data = digit_subset(train_data, BASE_DIGITS)
    if train_subset > 1:
        base_data = utils.data_subset(base_data, train_subset)
    loader = DataLoader(base_data, batch_size=batch_size, shuffle=True)
    return build_frozen_classifier(
        loader, encoder, hidden, len(BASE_DIGITS), epochs,
    )


def _teach_held_out_digit(
    net: StageModule, train_data: Dataset, encoder: SpikeEncoder,
) -> OneShotAssociativeMemory:
    """Build a memory module and teach it one digit-9 example."""
    memory = OneShotAssociativeMemory(in_features=hidden_size(net))
    teach_image, _ = digit_subset(train_data, (HELD_OUT_DIGIT,))[0]
    hidden_spikes, _ = hidden_trace(net, encoder.encode_image(teach_image))
    memory.teach(hidden_spikes[:, 0, :])
    return memory


def _label_mapper(
    memory: OneShotAssociativeMemory,
) -> Callable[[int], int]:
    """Return a function mapping a combined index back to a real digit.

    Base indices ``0..len(BASE_DIGITS)-1`` are already real digits;
    every taught memory class maps to the held-out digit, since this
    POC only ever teaches one.
    """
    def _map(index: int) -> int:
        return index if index < len(BASE_DIGITS) else HELD_OUT_DIGIT
    return _map


def _accuracy(
    net: StageModule,
    memory: OneShotAssociativeMemory,
    subset: Subset,
    encoder: SpikeEncoder,
    label_map: Callable[[int], int],
    batch_size: int,
) -> float:
    """Return percent accuracy of the combined net+memory prediction."""
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            preds = combined_predictions(net, memory, encoder.encode(images))
            mapped = torch.tensor([label_map(int(p)) for p in preds])
            correct += int((mapped == labels).sum())
            total += len(labels)
    return 100.0 * correct / max(total, 1)
