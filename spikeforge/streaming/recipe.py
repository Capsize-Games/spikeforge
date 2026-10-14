"""P2/P3: the CPU training recipe, evaluation metrics, and anomaly rule.

P2 trains a small ``sequence_mlp`` trunk on windowed spikes with the shared
:func:`~spikeforge.simulator.runner.run` temporal loop and surrogate-gradient
cross-entropy. P3 scores class accuracy / macro-F1 / balanced accuracy and an
additive one-class anomaly score (``1 - max softmax``) with a threshold rule
derived from the train score distribution.

No scikit-learn dependency: the metrics are implemented here with torch only,
so a headless core install can train and evaluate.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from spikeforge.streaming.encoding import encode_windows
from spikeforge.streaming.recipe_anomaly import (
    _precision_recall,
    anomaly_rule,
    anomaly_scores,
    anomaly_threshold,
    auroc,
    fit_threshold,
)
from spikeforge.streaming.recipe_config import TOPOLOGY, StreamTrainConfig
from spikeforge.streaming.recipe_metrics import class_metrics
from spikeforge.streaming.recipe_result import StreamTrainingResult
from spikeforge.streaming.recipe_training import (
    build_model,
    predict_logits,
    train_classifier,
)
from spikeforge.streaming.stream_source import StreamDataset
from spikeforge.topology.stage_module import StageModule
from spikeforge.tracking.seed import set_seed

__all__ = [
    "TOPOLOGY",
    "StreamTrainConfig",
    "StreamTrainingResult",
    "anomaly_rule",
    "anomaly_scores",
    "anomaly_threshold",
    "auroc",
    "build_model",
    "class_metrics",
    "encode_windows",
    "evaluate",
    "fit_threshold",
    "predict_logits",
    "set_seed",
    "train_classifier",
]


def _classification_report(
    module: StageModule, dataset: StreamDataset, config: StreamTrainConfig
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Return the class-metrics report and the raw logits it was scored on."""
    logits = predict_logits(module, dataset.windows, config)
    predictions = logits.argmax(dim=-1)
    report = class_metrics(dataset.labels, predictions, config.num_classes)
    report.pop("_flat", None)
    return report, logits


def _anomaly_report(
    logits: torch.Tensor, dataset: StreamDataset, threshold: Optional[float]
) -> Dict[str, Any]:
    """Return the anomaly-detection fields of the evaluation report."""
    scores = anomaly_scores(logits)
    resolved = (
        anomaly_threshold(scores) if threshold is None else float(threshold)
    )
    flags = anomaly_rule(scores, resolved)
    precision, recall = _precision_recall(flags, dataset.anomaly)
    return {
        "num_windows": len(dataset),
        "anomaly_threshold": resolved,
        "anomaly_auroc": auroc(scores, dataset.anomaly),
        "anomaly_precision": precision,
        "anomaly_recall": recall,
    }


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
    report, logits = _classification_report(module, dataset, config)
    report.update(_anomaly_report(logits, dataset, threshold))
    return report
