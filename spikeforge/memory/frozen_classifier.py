"""Train a small classifier, then freeze it for the memory POC."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.simulator.runner import run
from spikeforge.topology.registry import build_topology
from spikeforge.topology.stage_module import StageModule

#: Learning rate for the base classifier's Adam optimizer.
LEARNING_RATE = 1e-2


def build_frozen_classifier(
    loader: DataLoader,
    encoder: SpikeEncoder,
    hidden: int,
    num_classes: int,
    epochs: int = 1,
) -> StageModule:
    """Train ``fc_small`` on ``loader``, then permanently freeze it.

    Freezing with :meth:`~torch.nn.Module.requires_grad_` makes "no
    forgetting" an engineering guarantee: nothing downstream can ever
    run an optimizer step against these parameters again.
    """
    _, net = build_topology(
        "fc_small", {"hidden": hidden, "num_classes": num_classes},
    )
    _train(net, loader, encoder, epochs)
    net.requires_grad_(False)
    net.eval()
    return net


def _train(
    net: StageModule,
    loader: DataLoader,
    encoder: SpikeEncoder,
    epochs: int,
) -> None:
    """Run a plain supervised loop over rate-coded spikes."""
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in loader:
            _train_batch(net, optimizer, loss_fn, encoder, images, labels)


def _train_batch(
    net: StageModule,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    encoder: SpikeEncoder,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Run one forward/backward/step over a single batch."""
    trajectory = run(net, encoder.encode(images))
    loss = loss_fn(trajectory.logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def hidden_size(net: StageModule) -> int:
    """Return the width of ``fc1``, the frozen net's hidden stage.

    Used by callers that build a memory module sized to match, since
    ``fc_small`` always names its hidden linear stage ``fc1``.
    """
    weight_shape: Tuple[int, ...] = tuple(
        net.get_submodule("fc1").weight.shape
    )
    return weight_shape[0]
