"""The result of one ``prune`` call: pruned tensors plus their report."""

from dataclasses import dataclass
from typing import Mapping

import torch

from spikeforge.compression.pruning_report import PruningReport


@dataclass(frozen=True)
class PrunedWeights:
    """A pruned state dict paired with the report describing it."""

    tensors: Mapping[str, torch.Tensor]
    report: PruningReport
