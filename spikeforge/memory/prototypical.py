"""Prototypical-network embedding and loss over spiking readouts."""

from typing import Tuple

import torch

from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.simulator.runner import run
from spikeforge.topology.stage_module import StageModule


def embed(
    net: StageModule, encoder: SpikeEncoder, images: torch.Tensor,
) -> torch.Tensor:
    """Return ``[B, embed_dim]`` firing-rate embeddings for ``images``.

    The embedding is the readout's per-neuron mean firing rate over
    time (``Trajectory.logits``) -- the same quantity a fixed
    classifier would softmax over, repurposed here as a spike-pattern
    embedding rather than a class score.
    """
    return run(net, encoder.encode(images)).logits


def prototypical_loss(
    support_emb: torch.Tensor,
    support_labels: torch.Tensor,
    query_emb: torch.Tensor,
    query_labels: torch.Tensor,
    n_way: int,
) -> Tuple[torch.Tensor, float]:
    """Return ``(cross-entropy loss, accuracy)`` over prototype distances.

    Each class's prototype is its support embeddings' mean; a query
    is scored by negative distance to every prototype, so "close to
    the right prototype" is trained the same way a fixed classifier
    trains "high score for the right class".
    """
    prototypes = torch.stack(
        [
            support_emb[support_labels == c].mean(dim=0)
            for c in range(n_way)
        ]
    )
    distances = torch.cdist(query_emb, prototypes)
    loss = torch.nn.functional.cross_entropy(-distances, query_labels)
    accuracy = (distances.argmin(dim=1) == query_labels).float().mean()
    return loss, float(accuracy)
