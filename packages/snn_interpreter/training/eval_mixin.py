"""Held-out evaluation for the training engine."""

from itertools import islice
from typing import List, Optional, Tuple

import torch

from snn_interpreter.data.data_loader import build_loader
from snn_interpreter.observability import metrics

EVAL_EVERY = 5  # evaluate the held-out set every N steps
EVAL_BATCHES = 4  # number of test batches to score

_Batch = Tuple[torch.Tensor, torch.Tensor]


class EvalMixin:
    """Score held-out batches periodically during training."""

    _test_batches: Optional[List[_Batch]]

    def _load_test_batches(self) -> List[_Batch]:
        """Cache a few held-out batches for scoring."""
        if self._test_batches is None:
            loader = build_loader(self._dataset, 1, 1000, train=False)
            self._test_batches = list(islice(loader, EVAL_BATCHES))
        return self._test_batches

    def evaluate(self) -> float:
        """Return held-out accuracy over the cached test batches."""
        with metrics.timer("validation.seconds"):
            accuracy = self._accuracy()
        metrics.gauge("validation.accuracy", accuracy)
        metrics.counter("validation.runs")
        return accuracy

    def _accuracy(self) -> float:
        """Score the cached held-out batches and return accuracy."""
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in self._load_test_batches():
                correct += int(
                    (self.predict(inputs) == targets.to(self._device)).sum()
                )
                total += len(targets)
        return 100.0 * correct / max(total, 1)

    def _maybe_evaluate(self, step: int, total: int) -> Optional[float]:
        """Evaluate periodically and on the final step."""
        due = step == 1 or step == total or step % EVAL_EVERY == 0
        return self.evaluate() if due else None
