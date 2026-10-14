"""One-class anomaly scoring with a threshold rule over the same trunk."""

from typing import Tuple

import torch

from spikeforge.streaming.recipe_config import StreamTrainConfig
from spikeforge.streaming.recipe_training import predict_logits
from spikeforge.streaming.stream_source import StreamDataset
from spikeforge.topology.stage_module import StageModule


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
