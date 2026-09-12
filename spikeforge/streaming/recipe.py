"""P2/P3: the CPU training recipe, evaluation metrics, and anomaly rule.

P2 trains a small ``sequence_mlp`` trunk on windowed spikes with the shared
:func:`~spikeforge.simulator.runner.run` temporal loop and surrogate-gradient
cross-entropy. P3 scores class accuracy / macro-F1 / balanced accuracy and an
additive one-class anomaly score (``1 - max softmax``) with a threshold rule
derived from the train score distribution.

No scikit-learn dependency: the metrics are implemented here with torch only,
so a headless core install can train and evaluate.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch.nn.functional import cross_entropy

from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.simulator.runner import run
from spikeforge.streaming.encoding import encode_windows
from spikeforge.streaming.stream_source import StreamDataset
from spikeforge.topology import registry
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule
from spikeforge.tracking.seed import set_seed

#: Topology the recipe trains; the NIR-mappable sequence preset.
TOPOLOGY: str = "sequence_mlp"


@dataclass(frozen=True)
class StreamTrainConfig:
    """The geometry, encode contract, and optimiser for one training run."""

    seq_length: int = 16
    features: int = 4
    hidden: int = 16
    num_classes: int = 3
    num_steps: int = 6
    coding: str = "delta"
    delta_threshold: float = 4.0
    delta_along_window: bool = False
    lr: float = 5e-2
    epochs: int = 20
    batch_size: int = 8
    seed: int = 0

    def validate(self) -> None:
        """Raise :class:`ValueError` when the config is unusable."""
        if min(self.seq_length, self.features, self.hidden) < 1:
            raise ValueError("seq_length, features and hidden must be >= 1")
        if self.num_classes < 2 or self.num_steps < 1:
            raise ValueError(
                "num_classes >= 2 and num_steps >= 1 are required"
            )
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        self.encode_spec().validate()

    def encode_spec(self) -> EncodeSpec:
        """Return the frozen encode contract both train and serve share."""
        return EncodeSpec(
            coding=self.coding,
            num_steps=self.num_steps,
            delta_threshold=self.delta_threshold,
            random_seed=self.seed,
        )

    def topology_params(self) -> Dict[str, Any]:
        """Return the ``sequence_mlp`` preset parameters for this config."""
        return {
            "seq_length": int(self.seq_length),
            "features": int(self.features),
            "hidden": int(self.hidden),
            "num_classes": int(self.num_classes),
        }


@dataclass(frozen=True)
class StreamTrainingResult:
    """A trained trunk plus the contract needed to serve it."""

    spec: TopologySpec
    module: StageModule
    config: StreamTrainConfig
    encode_spec: EncodeSpec
    history: Tuple[Dict[str, float], ...]


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


def train_classifier(
    train: StreamDataset,
    config: StreamTrainConfig,
) -> StreamTrainingResult:
    """Train the trunk for ``config.epochs`` on ``train`` (CPU, deterministic).

    The shuffle generator is seeded once and is independent of the encode
    RNG, so two runs of the same config visit the same batches in the same
    order and produce the same loss history.
    """
    config.validate()
    set_seed(config.seed)
    spec, module = build_model(config)
    module.train()
    optimizer = torch.optim.Adam(module.parameters(), lr=config.lr)
    encode_spec = config.encode_spec()
    windows = train.windows
    labels = train.labels
    total = int(windows.size(0))
    order = torch.Generator().manual_seed(config.seed)
    history = []
    for epoch in range(config.epochs):
        permutation = torch.randperm(total, generator=order)
        epoch_loss = 0.0
        correct = 0
        batches = 0
        for start in range(0, total, config.batch_size):
            index = permutation[start:start + config.batch_size]
            spikes = encode_windows(
                windows[index],
                encode_spec,
                delta_along_window=config.delta_along_window,
            )
            logits = _mean_logits(module, spikes)
            loss = cross_entropy(logits, labels[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            correct += int(
                (logits.argmax(dim=-1) == labels[index]).sum().item()
            )
            batches += 1
        history.append(
            {
                "epoch": float(epoch),
                "loss": epoch_loss / max(batches, 1),
                "train_accuracy": correct / max(total, 1),
            }
        )
    module.eval()
    return StreamTrainingResult(
        spec=spec,
        module=module,
        config=config,
        encode_spec=encode_spec,
        history=tuple(history),
    )


def class_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> Dict[str, Any]:
    """Return accuracy, macro-F1, balanced accuracy, and per-class recall."""
    truth = torch.as_tensor(y_true, dtype=torch.long)
    pred = torch.as_tensor(y_pred, dtype=torch.long)
    total = int(truth.numel())
    if total == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "per_class_recall": [0.0] * int(num_classes),
        }
    confusion = torch.bincount(
        truth * int(num_classes) + pred, minlength=int(num_classes) ** 2
    ).reshape(int(num_classes), int(num_classes)).float()
    diag = torch.diagonal(confusion)
    accuracy = float(diag.sum().item() / total)
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    recall = diag / support.clamp_min(1.0)
    precision = diag / predicted.clamp_min(1.0)
    denom = (precision + recall).clamp_min(1e-12)
    f1 = 2.0 * precision * recall / denom
    present = support > 0
    macro_f1 = float(f1[present].mean().item()) if bool(present.any()) else 0.0
    balanced = (
        float(recall[present].mean().item()) if bool(present.any()) else 0.0
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced,
        "per_class_recall": [float(value) for value in recall.tolist()],
    }


def anomaly_scores(logits: torch.Tensor) -> torch.Tensor:
    """Return a one-class anomaly score per row of ``logits`` (``[B, C]``).

    The score is ``1 - max softmax``: an input the trunk cannot confidently
    place in any trained class scores high, which is exactly the additive
    one-class head the plan describes reusing the same temporal trunk.
    """
    probabilities = torch.softmax(
        torch.as_tensor(logits, dtype=torch.float32), dim=-1
    )
    return 1.0 - probabilities.max(dim=-1).values


def anomaly_threshold(
    scores: torch.Tensor, percentile: float = 95.0
) -> float:
    """Return the ``percentile``-th quantile of a train score distribution."""
    values = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values, float(percentile) / 100.0).item())


def anomaly_rule(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return the boolean anomaly flags for ``scores`` above ``threshold``."""
    values = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    return values > float(threshold)


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return the average (tie-corrected) rank of every value, 1-indexed."""
    count = values.numel()
    sorted_values, order = torch.sort(values)
    ranks_sorted = torch.arange(1, count + 1, dtype=torch.float32)
    start = 0
    while start < count:
        end = start
        while end + 1 < count and float(sorted_values[end + 1]) == float(
            sorted_values[start]
        ):
            end += 1
        ranks_sorted[start:end + 1] = 0.5 * (start + end + 2.0)
        start = end + 1
    ranks = torch.empty(count, dtype=torch.float32)
    ranks[order] = ranks_sorted
    return ranks


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the Mann-Whitney AUROC of ``scores`` against boolean labels.

    ``labels`` marks positives (anomalies). With only one class present the
    statistic is undefined and ``0.5`` is returned rather than a misleading 0
    or 1.
    """
    values = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    positive = torch.as_tensor(labels, dtype=torch.bool).reshape(-1)
    n_pos = float(positive.sum().item())
    n_neg = float((~positive).sum().item())
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.5
    ranks = _average_ranks(values)
    rank_sum = float(ranks[positive].sum().item())
    return (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def _precision_recall(
    predicted: torch.Tensor, truth: torch.Tensor
) -> Tuple[float, float]:
    """Return ``(precision, recall)`` for two boolean flag vectors."""
    pred = torch.as_tensor(predicted, dtype=torch.bool).reshape(-1)
    actual = torch.as_tensor(truth, dtype=torch.bool).reshape(-1)
    true_positive = float((pred & actual).sum().item())
    precision = true_positive / max(float(pred.sum().item()), 1.0)
    recall = true_positive / max(float(actual.sum().item()), 1.0)
    return precision, recall


def fit_threshold(
    module: StageModule,
    train: StreamDataset,
    config: StreamTrainConfig,
    percentile: float = 95.0,
) -> float:
    """Return the threshold fitted on the **train** score distribution."""
    logits = predict_logits(module, train.windows, config)
    return anomaly_threshold(anomaly_scores(logits), percentile)


def evaluate(
    module: StageModule,
    dataset: StreamDataset,
    config: StreamTrainConfig,
    *,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Return the classification + anomaly acceptance report for ``dataset``.

    When ``threshold`` is omitted the dataset's own score quantile is used as a
    convenience; a real deployment rule fits the threshold with
    :func:`fit_threshold` on the train split (as the tests and example do).
    """
    logits = predict_logits(module, dataset.windows, config)
    predictions = logits.argmax(dim=-1)
    report: Dict[str, Any] = class_metrics(
        dataset.labels, predictions, config.num_classes
    )
    report.pop("_flat", None)
    scores = anomaly_scores(logits)
    resolved = (
        anomaly_threshold(scores) if threshold is None else float(threshold)
    )
    flags = anomaly_rule(scores, resolved)
    precision, recall = _precision_recall(flags, dataset.anomaly)
    report.update(
        {
            "num_windows": len(dataset),
            "anomaly_threshold": resolved,
            "anomaly_auroc": auroc(scores, dataset.anomaly),
            "anomaly_precision": precision,
            "anomaly_recall": recall,
        }
    )
    return report
