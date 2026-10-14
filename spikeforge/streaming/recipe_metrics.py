"""Classification metrics over torch tensors (no scikit-learn)."""

from typing import Any, Dict, Tuple

import torch


def _empty_class_metrics(num_classes: int) -> Dict[str, Any]:
    """Return zeroed metrics when there are no samples."""
    return {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "per_class_recall": [0.0] * int(num_classes),
    }


def _confusion_matrix(
    truth: torch.Tensor, pred: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Return the ``[C, C]`` confusion matrix for ``truth``/``pred``."""
    counts = torch.bincount(
        truth * int(num_classes) + pred, minlength=int(num_classes) ** 2
    )
    return counts.reshape(int(num_classes), int(num_classes)).float()


def _precision_recall_f1(
    confusion: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:
    """Return recall, macro-F1, and balanced accuracy from ``confusion``."""
    diag = torch.diagonal(confusion)
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
    return recall, macro_f1, balanced


def class_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> Dict[str, Any]:
    """Return accuracy, macro-F1, balanced accuracy, and per-class recall."""
    truth = torch.as_tensor(y_true, dtype=torch.long)
    pred = torch.as_tensor(y_pred, dtype=torch.long)
    total = int(truth.numel())
    if total == 0:
        return _empty_class_metrics(num_classes)
    confusion = _confusion_matrix(truth, pred, num_classes)
    recall, macro_f1, balanced = _precision_recall_f1(confusion)
    accuracy = float(torch.diagonal(confusion).sum().item() / total)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced,
        "per_class_recall": [float(value) for value in recall.tolist()],
    }
