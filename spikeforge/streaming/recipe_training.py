"""Train the recipe's ``sequence_mlp`` trunk on windowed spikes (CPU)."""

from typing import Dict, Tuple

import torch
from torch.nn.functional import cross_entropy

from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.simulator.runner import run
from spikeforge.streaming.encoding import encode_windows
from spikeforge.streaming.recipe_config import TOPOLOGY, StreamTrainConfig
from spikeforge.streaming.recipe_result import StreamTrainingResult
from spikeforge.streaming.stream_source import StreamDataset
from spikeforge.topology import registry
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule
from spikeforge.tracking.seed import set_seed


def build_model(
    config: StreamTrainConfig,
) -> Tuple[TopologySpec, StageModule]:
    """Return ``(spec, module)`` for the recipe's ``sequence_mlp``."""
    return registry.build_topology(TOPOLOGY, config.topology_params())


def _mean_logits(module: StageModule, spikes: torch.Tensor) -> torch.Tensor:
    """Return ``[B, C]`` readout logits averaged over the window tokens."""
    trajectory = run(module, spikes)
    return trajectory.logits.mean(dim=1)


def predict_logits(
    module: StageModule,
    windows: torch.Tensor,
    config: StreamTrainConfig,
) -> torch.Tensor:
    """Return ``[B, C]`` mean logits for a batch of ``[B, L, D]`` windows."""
    module.eval()
    spikes = encode_windows(
        windows,
        config.encode_spec(),
        delta_along_window=config.delta_along_window,
    )
    with torch.no_grad():
        return _mean_logits(module, spikes)


def _train_one_batch(
    module: StageModule,
    optimizer: torch.optim.Optimizer,
    windows: torch.Tensor,
    labels: torch.Tensor,
    encode_spec: EncodeSpec,
    config: StreamTrainConfig,
) -> Tuple[float, int]:
    """Run one optimizer step on one batch; return (loss, correct count)."""
    spikes = encode_windows(
        windows, encode_spec, delta_along_window=config.delta_along_window
    )
    logits = _mean_logits(module, spikes)
    loss = cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    correct = int((logits.argmax(dim=-1) == labels).sum().item())
    return float(loss.item()), correct


def _train_one_epoch(
    module: StageModule, optimizer: torch.optim.Optimizer,
    train: StreamDataset, encode_spec: EncodeSpec,
    config: StreamTrainConfig, order: torch.Generator,
) -> Tuple[float, float]:
    """Run one training epoch; return its mean loss and accuracy."""
    total = int(train.windows.size(0))
    permutation = torch.randperm(total, generator=order)
    epoch_loss = 0.0
    correct = batches = 0
    for start in range(0, total, config.batch_size):
        index = permutation[start:start + config.batch_size]
        loss, hits = _train_one_batch(
            module, optimizer, train.windows[index], train.labels[index],
            encode_spec, config,
        )
        epoch_loss += loss
        correct += hits
        batches += 1
    return epoch_loss / max(batches, 1), correct / max(total, 1)


def _run_epochs(
    module: StageModule, optimizer: torch.optim.Optimizer,
    train: StreamDataset, encode_spec: EncodeSpec,
    config: StreamTrainConfig, order: torch.Generator,
) -> Tuple[Dict[str, float], ...]:
    """Run every training epoch; return the per-epoch metric history."""
    history = []
    for epoch in range(config.epochs):
        loss, accuracy = _train_one_epoch(
            module, optimizer, train, encode_spec, config, order
        )
        history.append(
            {"epoch": float(epoch), "loss": loss, "train_accuracy": accuracy}
        )
    return tuple(history)


def _finalize_training(
    spec: TopologySpec,
    module: StageModule,
    config: StreamTrainConfig,
    encode_spec: EncodeSpec,
    history: Tuple[Dict[str, float], ...],
) -> StreamTrainingResult:
    """Switch to eval mode and package the trained trunk for serving."""
    module.eval()
    return StreamTrainingResult(
        spec=spec,
        module=module,
        config=config,
        encode_spec=encode_spec,
        history=history,
    )


def train_classifier(
    train: StreamDataset,
    config: StreamTrainConfig,
) -> StreamTrainingResult:
    """Train the trunk for ``config.epochs`` on ``train`` (CPU, deterministic).

    The shuffle generator is seeded once, independent of the encode RNG, so
    two runs of the same config produce the same loss history.
    """
    config.validate()
    set_seed(config.seed)
    spec, module = build_model(config)
    module.train()
    optimizer = torch.optim.Adam(module.parameters(), lr=config.lr)
    encode_spec = config.encode_spec()
    order = torch.Generator().manual_seed(config.seed)
    history = _run_epochs(
        module, optimizer, train, encode_spec, config, order
    )
    return _finalize_training(spec, module, config, encode_spec, history)
